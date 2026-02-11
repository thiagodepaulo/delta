import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_
from delta.configs.caimira import CaimiraConfig
from delta.models.base import BaseDeltaModel
from delta.configs.ntm import NTMConfig
from typing import Any, Optional
from delta.models.caimira import CaimiraModel

class NTMModel(BaseDeltaModel):
    """
    Delta wrapper for torchScholar.

    Expected batch keys (tensors):
      - x:  [B, vocab_size] float (word counts / bow)
      - y:  [B, n_labels] float (one-hot) OR None if n_labels=0
      - pc: [B, n_prior_covars] float OR None if n_prior_covars=0
      - tc: [B, n_topic_covars] float OR None if n_topic_covars=0
    """

    def __init__(self, config: NTMConfig):
        super(NTMModel, self).__init__()        
        self.config = config        

        device = getattr(config, "device", None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_ = torch.device(device)
                
         # load the configuration
        self.vocab_size = config.vocab_size
        self.words_emb_dim = config.words_emb_dim
        self.n_topics = config.n_topics
        self.n_labels = config.n_labels
        self.n_prior_covars = config.n_prior_covars
        self.n_topic_covars = config.n_topic_covars
        self.classifier_layers = config.classifier_layers
        self.use_interactions = config.use_interactions
        self.l1_beta_reg = config.l1_beta_reg
        self.l1_beta_c_reg = config.l1_beta_c_reg
        self.l1_beta_ci_reg = config.l1_beta_ci_reg
        self.l2_prior_reg = config.l2_prior_reg 
        self.device = device
        self.classify_from_covars = config.classify_from_covars
        self.eta_bn_prop = config.eta_bn_prop
        self.alpha = config.alpha
        
        bg_init = None
        
         # interpret alpha as either a (symmetric) scalar prior or a vector prior
        if np.array(self.alpha).size == 1:
            # if alpha is a scalar, create a symmetric prior vector
            self.alpha = self.alpha * np.ones((1, self.config.n_topics)).astype(np.float32)
        else:
            # otherwise use the prior as given
            self.alpha = np.array(self.alpha).astype(np.float32)
            assert len(self.alpha) == self.config.n_topics
        
        # create a layer for prior covariates to influence the document prior
        if self.n_prior_covars > 0:
            self.prior_covar_weights = nn.Linear(self.n_prior_covars, self.n_topics, bias=False)
        else:
            self.prior_covar_weights = None
            
        self.criterion = nn.BCEWithLogitsLoss(reduction="none").to(self.device)            

        encoder = Encoder_1()
        self.classifier_input_dim = encoder.classifier_input_dim        
        
        # create the decoder
        self.beta_layer = nn.Linear(self.n_topics, self.vocab_size)

        xavier_uniform_(self.beta_layer.weight)
        
        if bg_init is not None:
            self.beta_layer.bias.data.copy_(torch.from_numpy(bg_init))
            self.beta_layer.bias.requires_grad = False
        self.beta_layer = self.beta_layer.to(self.device)

        if self.n_topic_covars > 0:
            self.beta_c_layer = nn.Linear(self.n_topic_covars, self.vocab_size, bias=False).to(self.device)
            if self.use_interactions:
                self.beta_ci_layer = nn.Linear(self.n_topics * self.n_topic_covars, self.vocab_size, bias=False).to(self.device)

        # create the classifier
        self.predictor = Predictor_1(config, self.classifier_input_dim)
        
        # create a final batchnorm layer
        self.eta_bn_layer = nn.BatchNorm1d(self.vocab_size, eps=0.001, momentum=0.001, affine=True).to(self.device)
        self.eta_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.vocab_size)).to(self.device))
        self.eta_bn_layer.weight.requires_grad = False

        # create the document prior terms
        prior_mean = (np.log(self.alpha).T - np.mean(np.log(self.alpha), 1)).T
        prior_var = (((1.0 / self.alpha) * (1 - (2.0 / self.n_topics))).T + (1.0 / (self.n_topics * self.n_topics)) * np.sum(1.0 / self.alpha, 1)).T

        prior_mean = np.array(prior_mean).reshape((1, self.n_topics))
        prior_logvar = np.array(np.log(prior_var)).reshape((1, self.n_topics))
        self.prior_mean = torch.from_numpy(prior_mean).to(self.device)
        self.prior_mean.requires_grad = False
        self.prior_logvar = torch.from_numpy(prior_logvar).to(self.device)
        self.prior_logvar.requires_grad = False

    def forward(self, batch, var_scale=1.0):               
                
        posterior_mean_bn, posterior_logvar_bn, TC = self.encoder(batch)
        
        posterior_var = posterior_logvar_bn.exp().to(self.device)
        
        # sample noise from a standard normal
        #eps = X.data.new().resize_as_(posterior_mean_bn.data).normal_().to(self.device)
        eps = torch.randn_like(posterior_mean_bn)

        # compute the sampled latent representation
        z = posterior_mean_bn + posterior_var.sqrt() * eps * var_scale
        z_do = self.z_dropout_layer(z)

        # pass the document representations through a softmax
        theta = F.softmax(z_do, dim=1)

        # combine latent representation with topics and background
        # beta layer here includes both the topic weights and the background term (as a bias)
        eta = self.beta_layer(theta)

        # add deviations for covariates (and interactions)
        if self.n_topic_covars > 0:
            eta = eta + self.beta_c_layer(TC)
            if self.use_interactions:
                theta_rsh = theta.unsqueeze(2)
                tc_emb_rsh = TC.unsqueeze(1)
                covar_interactions = theta_rsh * tc_emb_rsh
                batch_size, _, _ = covar_interactions.shape
                eta += self.beta_ci_layer(covar_interactions.reshape((batch_size, self.n_topics * self.n_topic_covars)))

        # pass the unnormalized word probabilities through a batch norm layer
        eta_bn = self.eta_bn_layer(eta)
        #eta_bn = eta

        # compute X recon with and without batchnorm on eta, and take a convex combination of them
        X_recon_bn = F.softmax(eta_bn, dim=1)
        X_recon_no_bn = F.softmax(eta, dim=1)
        X_recon = self.eta_bn_prop * X_recon_bn + (1.0 - self.eta_bn_prop) * X_recon_no_bn

        # predict labels        
        Y_recon = self.predictor(theta, TC, batch)
                 
        # compute the document prior if using prior covariates
        if self.n_prior_covars > 0:
            prior_mean = self.prior_covar_weights(PC)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        else:
            prior_mean   = self.prior_mean.expand_as(posterior_mean)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)

        if compute_loss:
            log = self._loss(X, Y, X_recon, Y_recon, prior_mean, prior_logvar, posterior_mean_bn, posterior_logvar_bn, do_average, l1_beta, l1_beta_c, l1_beta_ci)            
            return theta, X_recon, Y_recon, log
            # return theta, X_recon, Y_recon, self._loss(X, Y, X_recon, Y_recon, prior_mean, prior_logvar, posterior_mean_bn, posterior_logvar_bn, do_average, l1_beta, l1_beta_c, l1_beta_ci)
        else:
            return theta, X_recon, Y_recon

        
        theta, x_recon, y_recon, log_loss = out
        return {                                    
            "loss": log_loss["loss"],            
            "logits": y_recon,
            "nl_loss": log_loss["nl_loss"],
            "ll_loss": log_loss["ll_loss"],
            "kld_loss": log_loss["kld_loss"],
        }
                

    @torch.no_grad()
    def encode_theta(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Convenience: get theta without computing loss."""
        return self.forward(batch, compute_loss=False)["theta"]

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Convenience: get class probs (if n_labels>0)."""
        out = self.forward(batch, compute_loss=False)
        return out["y_recon"]
    
    def get_bg(self):
        """
        Return the background terms
        """
        bg = self.net.beta_layer.to('cpu').bias.detach().numpy()
        self.net.beta_layer.to(self.device_)
        return bg
    
    def get_weights(self):
        """
        Return the topic-vocabulary deviation weights
        """
        emb = self.net.beta_layer.to('cpu').weight.detach().numpy().T
        self.net.beta_layer.to(self.device_)
        return emb
                
    def get_covar_weights(self):
        """
        Return the topic weight (deviations) associated with the topic covariates
        """
        emb = self.net.beta_c_layer.to('cpu').weight.detach().numpy().T
        self.net.beta_c_layer.to(self.device_)
        return emb                

class Encoder_1(nn.Module):
    
    def __init__(self, config, update_embeddings=True, init_emb = None):
        super(Encoder_1, self).__init__()
        
        self.vocab_size = config.vocab_size
        self.words_emb_dim = config.words_emb_dim
        self.n_topics = config.n_topics
        self.n_labels = config.n_labels
        self.n_prior_covars = config.n_prior_covars
        self.n_topic_covars = config.n_topic_covars
        
        # create the encoder
        self.embeddings_x_layer = nn.Linear(self.vocab_size, self.words_emb_dim, bias=False)
        self.emb_size = self.words_emb_dim
        self.classifier_input_dim = self.n_topics
        if self.n_prior_covars > 0:
            self.emb_size += self.n_prior_covars            
            self.classifier_input_dim += self.n_prior_covars
        if self.n_topic_covars > 0:
            self.emb_size += self.n_topic_covars
            self.classifier_input_dim += self.n_topic_covars
        if self.n_labels > 0:
            self.emb_size += self.n_labels if self.n_labels > 1 else 2
            

        self.encoder_dropout_layer = nn.Dropout(p=0.2)

        if not update_embeddings:
            self.embeddings_x_layer.weight.requires_grad = False
        if init_emb is not None:
            self.embeddings_x_layer.weight.data.copy_(torch.from_numpy(init_emb))
        else:
            xavier_uniform_(self.embeddings_x_layer.weight)                  
        
        # create the mean and variance components of the VAE
        self.mean_layer = nn.Linear(self.emb_size, self.n_topics)
        self.logvar_layer = nn.Linear(self.emb_size, self.n_topics)

        self.mean_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics)))
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics)))
        self.logvar_bn_layer.weight.requires_grad = False

        self.z_dropout_layer = nn.Dropout(p=0.2)
        
    def forward(self, batch):
        
        # Pull + default missing optional tensors to empty tensors of correct shape.
        X = batch['answer_bow']
        Y = batch['answer_label']                
        TC = batch['features']
            
        # embed the word counts
        en0_x = self.embeddings_x_layer(X)
        encoder_parts = [en0_x]
        
        # append additional components to the encoder, if given
        #if self.n_prior_covars > 0:
        #    encoder_parts.append(PC)
        if self.n_topic_covars > 0:
            encoder_parts.append(TC)        
        if self.n_labels > 0:
            if self.n_labels == 1:
                Y_onehot = torch.cat([1-Y, Y], dim=1)  # convert binary labels to two-column one-hot
                encoder_parts.append(Y_onehot)
            else:
                encoder_parts.append(Y)

        if len(encoder_parts) > 1:
            en0 = torch.cat(encoder_parts, dim=1)
        else:
            en0 = en0_x                    

        encoder_output = F.softplus(en0)
        encoder_output_do = self.encoder_dropout_layer(encoder_output)

        # compute the mean and variance of the document posteriors
        posterior_mean = self.mean_layer(encoder_output_do)
        posterior_logvar = self.logvar_layer(encoder_output_do)

        posterior_mean_bn = self.mean_bn_layer(posterior_mean)
        posterior_logvar_bn = self.logvar_bn_layer(posterior_logvar)
        #posterior_mean_bn = posterior_mean
        #posterior_logvar_bn = posterior_logvar                        

        return posterior_mean_bn, posterior_logvar_bn, TC
    

class Predictor_1(nn.Module):
    
    def __init__(self, config, classifier_input_dim):
        super(Predictor_1, self).__init__()
        
        self.classifier_input_dim = classifier_input_dim
        self.n_labels = config.n_labels
        self.n_topic_covars = config.n_topic_covars
        self.classifier_layers = config.classifier_layers
        
        if self.classifier_layers == 0:
            self.classifier_layer_0 = nn.Linear(self.classifier_input_dim, self.n_labels)
        elif self.classifier_layers == 1:
            self.classifier_layer_0 = nn.Linear(self.classifier_input_dim, self.classifier_input_dim)
            self.classifier_layer_1 = nn.Linear(self.classifier_input_dim, self.n_labels)
        elif self.classifier_layers == 100:
            self.classifier_layer_0 = CaimiraModel(CaimiraConfig(
                n_dim=self.n_topics,
                n_users=self.n_topic_covars,
                n_dim_item_embed=self.classifier_input_dim,
                n_dim_user_embed=self.lassifier_input_dim,
            ))
        else:
            print("Error: classifier_layers > 1 not implemented")

    def forward(self, theta, TC, batch):
                
        classifier_input = torch.cat([theta, TC], dim=1)
        if self.classifier_layers == 0:
            Y_recon = self.classifier_layer_0(classifier_input)
        elif self.classifier_layers == 1:
            cls0 = self.classifier_layer_0(classifier_input)
            #cls0_sp = F.softplus(cls0)
            Y_recon = self.classifier_layer_1(cls0)
        elif self.classifier_layers == 100:
            #Caimira
            batch_caimira = {
                'u_id': batch['u_id'],
                'answer_emb': classifier_input,
            }
            out_caimira = self.classifier_layer_0(batch_caimira)
            Y_recon = out_caimira['logits'].unsqueeze(1)            
        else:
            print("Error: classifier_layers > 1 not implemented")
    
        return Y_recon
        
        

class torchScholar(nn.Module):

    def __init__(self, config, alpha, update_embeddings=True, init_emb=None, bg_init=None, device='cpu', classify_from_covars=True):
        super(torchScholar, self).__init__()

        # load the configuration
        self.vocab_size = config.vocab_size
        self.words_emb_dim = config.words_emb_dim
        self.n_topics = config.n_topics
        self.n_labels = config.n_labels
        self.n_prior_covars = config.n_prior_covars
        self.n_topic_covars = config.n_topic_covars
        self.classifier_layers = config.classifier_layers
        self.use_interactions = config.use_interactions
        self.l1_beta_reg = config.l1_beta_reg
        self.l1_beta_c_reg = config.l1_beta_c_reg
        self.l1_beta_ci_reg = config.l1_beta_ci_reg
        self.l2_prior_reg = config.l2_prior_reg 
        self.device = device
        self.classify_from_covars = classify_from_covars

        # create a layer for prior covariates to influence the document prior
        if self.n_prior_covars > 0:
            self.prior_covar_weights = nn.Linear(self.n_prior_covars, self.n_topics, bias=False)
        else:
            self.prior_covar_weights = None
            
        self.criterion = nn.BCEWithLogitsLoss(reduction="none").to(self.device)            

        # create the encoder
        self.embeddings_x_layer = nn.Linear(self.vocab_size, self.words_emb_dim, bias=False)
        emb_size = self.words_emb_dim
        classifier_input_dim = self.n_topics
        if self.n_prior_covars > 0:
            emb_size += self.n_prior_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_prior_covars
        if self.n_topic_covars > 0:
            emb_size += self.n_topic_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_topic_covars
        if self.n_labels > 0:
            emb_size += self.n_labels if self.n_labels > 1 else 2
            

        self.encoder_dropout_layer = nn.Dropout(p=0.2)

        if not update_embeddings:
            self.embeddings_x_layer.weight.requires_grad = False
        if init_emb is not None:
            self.embeddings_x_layer.weight.data.copy_(torch.from_numpy(init_emb)).to(self.device)
        else:
            xavier_uniform_(self.embeddings_x_layer.weight)
        
        # create the mean and variance components of the VAE
        self.mean_layer = nn.Linear(emb_size, self.n_topics)
        self.logvar_layer = nn.Linear(emb_size, self.n_topics)

        self.mean_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.logvar_bn_layer.weight.requires_grad = False

        self.z_dropout_layer = nn.Dropout(p=0.2)

        # create the decoder
        self.beta_layer = nn.Linear(self.n_topics, self.vocab_size)

        xavier_uniform_(self.beta_layer.weight)
        if bg_init is not None:
            self.beta_layer.bias.data.copy_(torch.from_numpy(bg_init))
            self.beta_layer.bias.requires_grad = False
        self.beta_layer = self.beta_layer.to(self.device)

        if self.n_topic_covars > 0:
            self.beta_c_layer = nn.Linear(self.n_topic_covars, self.vocab_size, bias=False).to(self.device)
            if self.use_interactions:
                self.beta_ci_layer = nn.Linear(self.n_topics * self.n_topic_covars, self.vocab_size, bias=False).to(self.device)

        # create the classifier
        if self.n_labels > 0:
            if self.classifier_layers == 0:
                self.classifier_layer_0 = nn.Linear(classifier_input_dim, self.n_labels).to(self.device)
            elif self.classifier_layers == 1:
                self.classifier_layer_0 = nn.Linear(classifier_input_dim, classifier_input_dim).to(self.device)
                self.classifier_layer_1 = nn.Linear(classifier_input_dim, self.n_labels).to(self.device)
            elif self.classifier_layers == 100:
                self.classifier_layer_0 = CaimiraModel(CaimiraConfig(
                    n_dim=self.n_topics,
                    n_users=self.n_topic_covars,
                    n_dim_item_embed=classifier_input_dim,
                    n_dim_user_embed=classifier_input_dim,
                )).to(self.device)
            else:
                print("Error: classifier_layers > 1 not implemented")

        # create a final batchnorm layer
        self.eta_bn_layer = nn.BatchNorm1d(self.vocab_size, eps=0.001, momentum=0.001, affine=True).to(self.device)
        self.eta_bn_layer.weight.data.copy_(torch.from_numpy(np.ones(self.vocab_size)).to(self.device))
        self.eta_bn_layer.weight.requires_grad = False

        # create the document prior terms
        prior_mean = (np.log(alpha).T - np.mean(np.log(alpha), 1)).T
        prior_var = (((1.0 / alpha) * (1 - (2.0 / self.n_topics))).T + (1.0 / (self.n_topics * self.n_topics)) * np.sum(1.0 / alpha, 1)).T

        prior_mean = np.array(prior_mean).reshape((1, self.n_topics))
        prior_logvar = np.array(np.log(prior_var)).reshape((1, self.n_topics))
        self.prior_mean = torch.from_numpy(prior_mean).to(self.device)
        self.prior_mean.requires_grad = False
        self.prior_logvar = torch.from_numpy(prior_logvar).to(self.device)
        self.prior_logvar.requires_grad = False

    def forward(self, X, Y, PC, TC, u_id = None, compute_loss=True, do_average=True, eta_bn_prop=1.0, var_scale=1.0, l1_beta=None, l1_beta_c=None, l1_beta_ci=None):
        """
        Do a forward pass of the model
        :param X: np.array of word counts [batch_size x vocab_size]
        :param Y: np.array of labels [batch_size x n_classes]
        :param PC: np.array of covariates influencing the prior [batch_size x n_prior_covars]
        :param TC: np.array of covariates with explicit topic deviations [batch_size x n_topic_covariates]
        :param compute_loss: if True, compute and return the loss
        :param do_average: if True, average the loss over the minibatch
        :param eta_bn_prop: (float) a weight between 0 and 1 to interpolate between using and not using the final batchnorm layer
        :param var_scale: (float) a parameter which can be used to scale the variance of the random noise in the VAE
        :param l1_beta: np.array of prior variances for the topic weights
        :param l1_beta_c: np.array of prior variances on topic covariate deviations
        :param l1_beta_ci: np.array of prior variances on topic-covariate interactions
        :return: document representation; reconstruction; label probs; (loss, if requested)
        """
        
        # embed the word counts
        en0_x = self.embeddings_x_layer(X)
        encoder_parts = [en0_x]
        
        # append additional components to the encoder, if given
        if self.n_prior_covars > 0:
            encoder_parts.append(PC)
        if self.n_topic_covars > 0:
            encoder_parts.append(TC)        
        if self.n_labels > 0:
            if self.n_labels == 1:
                Y_onehot = torch.cat([1-Y, Y], dim=1)  # convert binary labels to two-column one-hot
                encoder_parts.append(Y_onehot)
            else:
                encoder_parts.append(Y)

        if len(encoder_parts) > 1:
            en0 = torch.cat(encoder_parts, dim=1).to(self.device)
        else:
            en0 = en0_x                    

        encoder_output = F.softplus(en0)
        encoder_output_do = self.encoder_dropout_layer(encoder_output)

        # compute the mean and variance of the document posteriors
        posterior_mean = self.mean_layer(encoder_output_do)
        posterior_logvar = self.logvar_layer(encoder_output_do)

        posterior_mean_bn = self.mean_bn_layer(posterior_mean)
        posterior_logvar_bn = self.logvar_bn_layer(posterior_logvar)
        #posterior_mean_bn = posterior_mean
        #posterior_logvar_bn = posterior_logvar

        posterior_var = posterior_logvar_bn.exp().to(self.device)

        # sample noise from a standard normal
        eps = X.data.new().resize_as_(posterior_mean_bn.data).normal_().to(self.device)

        # compute the sampled latent representation
        z = posterior_mean_bn + posterior_var.sqrt() * eps * var_scale
        z_do = self.z_dropout_layer(z)

        # pass the document representations through a softmax
        theta = F.softmax(z_do, dim=1)

        # combine latent representation with topics and background
        # beta layer here includes both the topic weights and the background term (as a bias)
        eta = self.beta_layer(theta)

        # add deviations for covariates (and interactions)
        if self.n_topic_covars > 0:
            eta = eta + self.beta_c_layer(TC)
            if self.use_interactions:
                theta_rsh = theta.unsqueeze(2)
                tc_emb_rsh = TC.unsqueeze(1)
                covar_interactions = theta_rsh * tc_emb_rsh
                batch_size, _, _ = covar_interactions.shape
                eta += self.beta_ci_layer(covar_interactions.reshape((batch_size, self.n_topics * self.n_topic_covars)))

        # pass the unnormalized word probabilities through a batch norm layer
        eta_bn = self.eta_bn_layer(eta)
        #eta_bn = eta

        # compute X recon with and without batchnorm on eta, and take a convex combination of them
        X_recon_bn = F.softmax(eta_bn, dim=1)
        X_recon_no_bn = F.softmax(eta, dim=1)
        X_recon = eta_bn_prop * X_recon_bn + (1.0 - eta_bn_prop) * X_recon_no_bn

        # predict labels        
        classifier_input = torch.cat([theta, TC], dim=1).to(self.device)
        if self.classifier_layers == 0:
            Y_recon = self.classifier_layer_0(classifier_input)
        elif self.classifier_layers == 1:
            cls0 = self.classifier_layer_0(classifier_input)
            #cls0_sp = F.softplus(cls0)
            Y_recon = self.classifier_layer_1(cls0)
        elif self.classifier_layers == 100:
            #Caimira
            batch = {
                'u_id': u_id,
                'answer_emb': classifier_input,
            }
            out_caimira = self.classifier_layer_0(batch)
            Y_recon = out_caimira['logits'].unsqueeze(1)            
        else:
            print("Error: classifier_layers > 1 not implemented")
        
         
        # compute the document prior if using prior covariates
        if self.n_prior_covars > 0:
            prior_mean = self.prior_covar_weights(PC)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        else:
            prior_mean   = self.prior_mean.expand_as(posterior_mean)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)

        if compute_loss:
            log = self._loss(X, Y, X_recon, Y_recon, prior_mean, prior_logvar, posterior_mean_bn, posterior_logvar_bn, do_average, l1_beta, l1_beta_c, l1_beta_ci)            
            return theta, X_recon, Y_recon, log
            # return theta, X_recon, Y_recon, self._loss(X, Y, X_recon, Y_recon, prior_mean, prior_logvar, posterior_mean_bn, posterior_logvar_bn, do_average, l1_beta, l1_beta_c, l1_beta_ci)
        else:
            return theta, X_recon, Y_recon
        
    
    
    def _loss(self, X, Y, X_recon, Y_recon, prior_mean, prior_logvar, posterior_mean, posterior_logvar, do_average=True, l1_beta=None, l1_beta_c=None, l1_beta_ci=None):        
        # compute reconstruction loss
        NL = -(X * (X_recon+1e-10).log()).sum(1)        
        
        # compute label loss
        if self.n_labels == 1:
            Y_onehot = torch.cat([1-Y, Y], dim=1)  # convert binary labels to two-column one-hot
            probs = torch.sigmoid(Y_recon).clamp(1e-10, 1-1e-10)
            Y_recon_onehot = torch.cat([probs, 1-probs], dim=1)
            NL += -(Y_onehot * (Y_recon_onehot+1e-10).log()).sum(1)
            
            
        # compute label loss
        LL = self.criterion(Y_recon, Y).sum(1)
        
        # compute KLD
        prior_var = prior_logvar.exp()
        posterior_var = posterior_logvar.exp()
        var_division    = posterior_var / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar

        # put KLD together
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.n_topics)
        
        loss = NL + LL + KLD
        
         # add regularization on prior
        if self.l2_prior_reg > 0 and self.n_prior_covars > 0:
            loss += self.l2_prior_reg * torch.pow(self.prior_covar_weights.weight, 2).sum()

        # add regularization on topic and topic covariate weights
        if self.l1_beta_reg > 0 and l1_beta is not None:
            l1_strengths_beta = torch.from_numpy(l1_beta).to(self.device)
            beta_weights_sq = torch.pow(self.beta_layer.weight, 2)
            loss += self.l1_beta_reg * (l1_strengths_beta * beta_weights_sq).sum()

        if self.n_topic_covars > 0 and l1_beta_c is not None and self.l1_beta_c_reg > 0:
            l1_strengths_beta_c = torch.from_numpy(l1_beta_c).to(self.device)
            beta_c_weights_sq = torch.pow(self.beta_c_layer.weight, 2)
            loss += self.l1_beta_c_reg * (l1_strengths_beta_c * beta_c_weights_sq).sum()

        if self.n_topic_covars > 0 and self.use_interactions and l1_beta_c is not None and self.l1_beta_ci_reg > 0:
            l1_strengths_beta_ci = torch.from_numpy(l1_beta_ci).to(self.device)
            beta_ci_weights_sq = torch.pow(self.beta_ci_layer.weight, 2)
            loss += self.l1_beta_ci_reg * (l1_strengths_beta_ci * beta_ci_weights_sq).sum()                
        
        log = {
            "loss": loss.mean() if do_average else loss,
            "nl_loss": NL.mean() if do_average else NL,
            "ll_loss": LL.mean() if do_average else LL,
            "kld_loss": KLD.mean() if do_average else KLD,
        }                
        
        return log
    
        
    
    def _loss2(self, X, Y, X_recon, Y_recon, prior_mean, prior_logvar, posterior_mean, posterior_logvar, do_average=True, l1_beta=None, l1_beta_c=None, l1_beta_ci=None):
        reco_loss = self._reco_loss(X, X_recon, do_average)
        label_loss = self._label_loss(Y, Y_recon, do_average)
        kld_loss = self._kld_loss(prior_mean, prior_logvar, posterior_mean, posterior_logvar, do_average)
        reg_loss = self._regularization_loss(l1_beta, l1_beta_c, l1_beta_ci)                
        
        total_loss = reco_loss + label_loss + kld_loss + reg_loss
        
        log = {
            "loss": total_loss,
            "recon_loss": reco_loss + kld_loss,
            "label_loss": label_loss,
            "kld_loss": kld_loss,
            "reg_loss": reg_loss,
        }

        return log

    def _loss_old(self, X, Y, X_recon, Y_recon, prior_mean, prior_logvar, posterior_mean, posterior_logvar, do_average=True, l1_beta=None, l1_beta_c=None, l1_beta_ci=None):

        # compute reconstruction loss
        NL = -(X * (X_recon+1e-10).log()).sum(1)
        # compute label loss
        if self.n_labels > 0:
            NL += -(Y * (Y_recon+1e-10).log()).sum(1)

        # compute KLD
        prior_var = prior_logvar.exp()
        posterior_var = posterior_logvar.exp()
        var_division    = posterior_var / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar

        # put KLD together
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.n_topics)

        # combine
        loss = (NL + KLD)

        # add regularization on prior
        if self.l2_prior_reg > 0 and self.n_prior_covars > 0:
            loss += self.l2_prior_reg * torch.pow(self.prior_covar_weights.weight, 2).sum()

        # add regularization on topic and topic covariate weights
        if self.l1_beta_reg > 0 and l1_beta is not None:
            l1_strengths_beta = torch.from_numpy(l1_beta).to(self.device)
            beta_weights_sq = torch.pow(self.beta_layer.weight, 2)
            loss += self.l1_beta_reg * (l1_strengths_beta * beta_weights_sq).sum()

        if self.n_topic_covars > 0 and l1_beta_c is not None and self.l1_beta_c_reg > 0:
            l1_strengths_beta_c = torch.from_numpy(l1_beta_c).to(self.device)
            beta_c_weights_sq = torch.pow(self.beta_c_layer.weight, 2)
            loss += self.l1_beta_c_reg * (l1_strengths_beta_c * beta_c_weights_sq).sum()

        if self.n_topic_covars > 0 and self.use_interactions and l1_beta_c is not None and self.l1_beta_ci_reg > 0:
            l1_strengths_beta_ci = torch.from_numpy(l1_beta_ci).to(self.device)
            beta_ci_weights_sq = torch.pow(self.beta_ci_layer.weight, 2)
            loss += self.l1_beta_ci_reg * (l1_strengths_beta_ci * beta_ci_weights_sq).sum()

        # average losses if desired
        if do_average:
            return loss.mean(), NL.mean(), KLD.mean()
        else:
            return loss, NL, KLD

    def predict_from_theta(self, theta, PC, TC):
        # Predict labels from a distribution over topics
        Y_recon = None
        if self.n_labels > 0:

            classifier_inputs = [theta]
            if self.classify_from_covars:
                if self.n_prior_covars > 0:
                    classifier_inputs.append(PC)
                if self.n_topic_covars > 0:
                    classifier_inputs.append(TC)
            if len(classifier_inputs) > 1:
                classifier_input = torch.cat(classifier_inputs, dim=1).to(self.device)
            else:
                classifier_input = theta.to(self.device)

            if self.classifier_layers == 0:
                decoded_y = self.classifier_layer_0(classifier_input)
            elif self.classifier_layers == 1:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                decoded_y = self.classifier_layer_1(cls0_sp)
            else:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                cls1 = self.classifier_layer_1(cls0_sp)
                cls1_sp = F.softplus(cls1)
                decoded_y = self.classifier_layer_1(cls1_sp)
            Y_recon = F.softmax(decoded_y, dim=1)

        return Y_recon
    
    
def print_weights(options, model, vocab, prior_covar_names=None, topic_covar_names=None):

    # print background
    bg = model.get_bg()
    if not options.no_bg:
        print_top_bg(bg, vocab)

    # print topics
    emb = model.get_weights()
    print("Topics:")
    maw, sparsity = print_top_words(emb, vocab)
    print("sparsity in topics = %0.4f" % sparsity)


    if prior_covar_names is not None:
        prior_weights = model.get_prior_weights()
        print("Topic prior associations:")
        print("Covariates:", ' '.join(prior_covar_names))
        for k in range(options.n_topics):
            output = str(k) + ': '
            for c in range(len(prior_covar_names)):
                output += '%.4f ' % prior_weights[c, k]
            print(output)
        if options.output_dir is not None:
            np.savez(os.path.join(options.output_dir, 'prior_w.npz'), weights=prior_weights, names=prior_covar_names)

    if topic_covar_names is not None:
        beta_c = model.get_covar_weights()
        print("Covariate deviations:")
        maw, sparsity = print_top_words(beta_c, vocab, topic_covar_names)
        print("sparsity in covariates = %0.4f" % sparsity)
        if options.output_dir is not None:
            np.savez(os.path.join(options.output_dir, 'beta_c.npz'), beta=beta_c, names=topic_covar_names)

        if options.interactions:
            print("Covariate interactions")
            beta_ci = model.get_covar_interaction_weights()
            print(beta_ci.shape)
            if topic_covar_names is not None:
                names = [str(k) + ':' + c for k in range(options.n_topics) for c in topic_covar_names]
            else:
                names = None
            maw, sparsity = print_top_words(beta_ci, vocab, names)
            if options.output_dir is not None:
                np.savez(os.path.join(options.output_dir, 'beta_ci.npz'), beta=beta_ci, names=names)
            print("sparsity in covariate interactions = %0.4f" % sparsity)


def print_top_words(beta, feature_names, topic_names=None, n_pos=8, n_neg=8, sparsity_threshold=1e-5, values=False):
    """
    Display the highest and lowest weighted words in each topic, along with mean ave weight and sparisty
    """
    sparsity_vals = []
    maw_vals = []
    for i in range(len(beta)):
        # sort the beta weights
        order = list(np.argsort(beta[i]))
        order.reverse()
        output = ''
        # get the top words
        for j in range(n_pos):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + ' '
                if values:
                    output += '(' + str(beta[i][order[j]]) + ') '

        order.reverse()
        if n_neg > 0:
            output += ' / '
        # get the bottom words
        for j in range(n_neg):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + ' '
                if values:
                    output += '(' + str(beta[i][order[j]]) + ') '

        # compute sparsity
        sparsity = float(np.sum(np.abs(beta[i]) < sparsity_threshold) / float(len(beta[i])))
        maw = np.mean(np.abs(beta[i]))
        sparsity_vals.append(sparsity)
        maw_vals.append(maw)
        output += '; sparsity=%0.4f' % sparsity

        # print the topic summary
        if topic_names is not None:
            output = topic_names[i] + ': ' + output
        else:
            output = str(i) + ': ' + output
        print(output)

    # return mean average weight and sparsity
    return np.mean(maw_vals), np.mean(sparsity_vals)
    
def print_top_bg(bg, feature_names, n_top_words=10):
    # Print the most highly weighted words in the background log frequency
    print('Background frequencies of top words:')
    print(" ".join([feature_names[j]
                    for j in bg.argsort()[:-n_top_words - 1:-1]]))
    temp = bg.copy()
    temp.sort()
    print(np.exp(temp[:-n_top_words-1:-1]))
    