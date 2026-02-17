import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import argparse
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from delta.lit_module import PrefModule, PrefDataModule
from delta.utils.config_utils import load_config
from delta.configs.trainer import TrainerConfig
from delta.reward_models.zero_rw import ZeroRWModel
from delta.reward_models.map_rw import MapRWModel
from delta.callbacks import BetaCallBack, PrintNTMTopics
from delta.models.ntm import NTMModel
from delta.configs.ntm import NTMConfig
from delta.models.ntm import print_weights
from types import SimpleNamespace


    
def main(args):
    parser = argparse.ArgumentParser(description="Train a model")    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--dts_name", type=str, required=True, help="Dataset name")    
    parser.add_argument("--dts_config_file", type=str, required=False, default='/home/thiagodepaulo/exp/delta/configs/data_configs.yaml', help="Dataset Config file name")
    parser.add_argument("--config_file", type=str, default='/home/thiagodepaulo/exp/delta/configs/exp_config.yaml', help="YAML config file")
    parser.add_argument("--model_name", type=str, default='model1', help="Model name to use")
    parser.add_argument("--rw_model_name", type=str, default='zero_rw', help="Reward model name to use")
    parser.add_argument("--exp_name", type=str, default="default_exp", help="Experiment name for logging")
    parser.add_argument("--exp_version", type=str, default="v0", help="Experiment version for logging")
    parser.add_argument("--features", type=str, default=None, help="Feature columns to use, 'all' for all features")
    parser.add_argument("--patience", type=int, default=10**9, help="Early stopping patience")
    parser.add_argument("--n_dim", type=int, default=50, help="Number of dimensions")
    parser.add_argument("--has_bow", action='store_true', default=False, help="Whether to use BOW embeddings")
    parser.add_argument("--print_topics", action='store_true', default=True, help="Whether to print NTM topics during training")
    parser.add_argument("--print_topics_every_n_epochs", type=int, default=1, help="How often to print NTM topics during training (in epochs)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    args = parser.parse_args()
        
            
    config_dict = load_config(args.config_file)
    trainer_config = TrainerConfig(**config_dict["trainer"])    
        
    args.batch_size = trainer_config.batch_size  # Ensure batch size is consistent
    args.has_bow = True
    
    data_module = PrefDataModule(args)    
    data_module.setup('fit')  # Ensure datasets are loaded before creating the model, in case vocab is needed for NTM
    
    rwModel = create_rw_model(args, config_dict)            
    model_instance = create_model(args, config_dict).to("cpu")    
    
    model = PrefModule(trainer_config, model_instance, rwModel)    
    lit_model = PrefModule.load_from_checkpoint("checkpoints/best-v9.ckpt",
                                                trainer_config=trainer_config,
                                                delta_model=model_instance,
                                                r_star_model=rwModel,
                                                map_location="cpu",   # IMPORTANT
                                                )
    lit_model.to("cpu")
    lit_model.delta_model.to("cpu")
    lit_model.delta_model.encoder.to("cpu")
    lit_model.eval()
    data_module.setup(stage="test")  # important!
    test_loader = data_module.test_dataloader()[0]

    #batch = next(iter(test_loader))    
    
    options = SimpleNamespace(
                no_bg=True,
                n_topics=lit_model.delta_model.n_topics,
                output_dir=None,
                interactions=False,
            )
    print_weights(options, lit_model.delta_model, data_module.args.vocab, topic_covar_names=getattr(lit_model.delta_model.config, 'topic_covar_names', None))

    #test_dts = data_module.test_dataloader()[0]    
    #batch = test_dts.dataset
    # answer_emb = X0['chosen_emb']
    #batch_wl = model. split_batch_for_pref(batch)
    #theta = model.delta_model.encode_theta(batch_wl['chosen'])[0].unsqueeze(0)
    #TC = batch_wl['chosen']['features'][0].unsqueeze(0)
    #u_id = batch_wl['chosen']['u_id'][0].item()
    
    #answer_emb = torch.cat([theta, TC], dim=1)
    
    n_topics = lit_model.delta_model.n_topics
    
    x_l_w = {}
    p_l_w = {}
    
    x_l_l = {}
    p_l_l = {}
    for batch in test_loader:
        batch_wl = model. split_batch_for_pref(batch)
        u_ids_w = batch_wl['chosen']['u_id']
        u_ids_l = batch_wl['rejected']['u_id']
        theta_w = model.delta_model.encode_theta(batch_wl['chosen'])
        theta_l = model.delta_model.encode_theta(batch_wl['rejected'])
        TC_w = batch_wl['chosen']['features']
        TC_l = batch_wl['rejected']['features']
        
        for u_id, theta, TC in zip(u_ids_w, theta_w, TC_w):            
            u_id = u_id.item()            
            #answer_emb = torch.cat([theta.unsqueeze(0), TC.unsqueeze(0)], dim=1)
            answer_emb = theta.unsqueeze(0)  #torch.cat([theta.unsqueeze(0), TC.unsqueeze(0)], dim=1)            
            xlk = []
            plk = []
            for k in range(n_topics):  # top 5 tópicos
                x, p, r, d = topic_curve_for_item(lit_model.delta_model.predictor.caimira, u_id, answer_emb, k)
                xlk.append(x)
                plk.append(p)
            if u_id not in x_l_w:
                x_l_w[u_id] = []
                p_l_w[u_id] = []
            x_l_w[u_id].append(xlk)
            p_l_w[u_id].append(plk)
        for u_id, theta, TC in zip(u_ids_l, theta_l, TC_l):
            u_id = u_id.item()
            #answer_emb = torch.cat([theta.unsqueeze(0), TC.unsqueeze(0)], dim=1)
            answer_emb = theta.unsqueeze(0)  #torch.cat([theta.unsqueeze(0), TC.unsqueeze(0)], dim=1)
            xlk = []
            plk = []
            for k in range(n_topics):  # top 5 tópicos
                x, p, r, d = topic_curve_for_item(lit_model.delta_model.predictor.caimira, u_id, answer_emb, k)
                xlk.append(x)
                plk.append(p)
            if u_id not in x_l_l:
                x_l_l[u_id] = []
                p_l_l[u_id] = []
            x_l_l[u_id].append(xlk)
            p_l_l[u_id].append(plk)
        
    
    for u_id in x_l_w.keys():
        for k in range(n_topics):  # top 5 tópicos       
            x_w = np.array([ v for subll in x_l_w[u_id] for v in subll[k] ])
            p_w = np.array([ v for subll in p_l_w[u_id] for v in subll[k] ])
            x_l = np.array([ v for subll in x_l_l[u_id] for v in subll[k] ])
            p_l = np.array([ v for subll in p_l_l[u_id] for v in subll[k] ])
            
            unique_x = sorted(set(list(x_w) + list(x_l)))
            mean_by_x_w = {xi: { 
                            "mean": p_w[x_w == xi].mean() if len(p_w[x_w == xi]) > 0 else 0,
                            "std": p_w[x_w == xi].std() if len(p_w[x_w == xi]) > 0 else 0,
                        }
                        for xi in unique_x}
            mean_by_x_l = {xi: { 
                            "mean": p_l[x_l == xi].mean() if len(p_l[x_l == xi]) > 0 else 0,
                            "std": p_l[x_l == xi].std() if len(p_l[x_l == xi]) > 0 else 0,
                        }
                        for xi in unique_x}
            
            plt.figure()            
            xs = unique_x
            means_w = [v["mean"] for v in mean_by_x_w.values()]
            stds_w = [v["std"] for v in mean_by_x_w.values()]
            means_l = [v["mean"] for v in mean_by_x_l.values()]
            stds_l = [v["std"] for v in mean_by_x_l.values()]
            plt.plot(xs, means_w, color="tab:blue", label="chosen")
            plt.fill_between(xs,
                 np.array(means_w) - np.array(stds_w),
                 np.array(means_w) + np.array(stds_w),
                 color="tab:blue",
                 alpha=0.3)
            plt.plot(xs, means_l, color="tab:orange", label="rejected")
            plt.fill_between(xs,
                 np.array(means_l) - np.array(stds_l),
                 np.array(means_l) + np.array(stds_l),
                 color="tab:orange",
                 alpha=0.3)                 
            plt.ylim(-0.02, 1.02)
            plt.xlabel(f"Habilidade no tópico k={k} (user {u_id})")
            plt.ylabel("P(Y=1)")
            plt.title(f"Curva por tópico (k={k}) user {u_id}")
            #plt.show()
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"plots/plot_{u_id}_{k}.png")
            plt.close()
            

    


def create_rw_model(args, config_dict=None):
    if args.rw_model_name == 'map_rw':
        rwModel = MapRWModel(args.map_rw_dataset)
    elif args.rw_model_name == 'zero_rw':
        rwModel = ZeroRWModel()
    else:
        raise ValueError(f"Unknown reward model name: {args.rw_model_name}")
    return rwModel

def create_model(args, config_dict):
    if args.model_name == 'model1':
        from delta.models.model1 import Model1
        from delta.configs.model1 import Model1Config

        model1_config = Model1Config(**config_dict["model1"])    
        model1_config.has_user_features = True if args.features is not None else False
        model1_config.n_dim_user_features = 45 if args.features == "all" else 0        
        model1_config.n_dim = args.n_dim
        model_instance = Model1(model1_config)
    elif args.model_name == 'caimira':
        from delta.configs.caimira import CaimiraConfig
        from delta.models.caimira import CaimiraModel
        
        caimira_config = CaimiraConfig(**config_dict["caimira"])
        caimira_config.n_dim = args.n_dim
        model_instance = CaimiraModel(caimira_config)  
    elif args.model_name == 'model2':
        from delta.models.model2 import Model2
        from delta.configs.model2 import Model2Config

        model2_config = Model2Config(**config_dict["model2"])
        model2_config.n_dim = args.n_dim
        model_instance = Model2(model2_config)
    elif args.model_name == 'ntm':        

        ntm_config = NTMConfig(**config_dict["ntm"])   
        ntm_config.device = "cpu"
        ntm_config.vocab_size = args.vocab_size  # Set vocab size from loaded vocab     
        ntm_config.n_topic_covars = args.n_features
        ntm_config.topic_covar_names = args.feature_columns        
        model_instance = NTMModel(ntm_config)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    
    return model_instance

@torch.no_grad()
def topic_curve_for_item(model, u_id: int, answer_emb: torch.Tensor, k: int,
                         x_min=-4.0, x_max=4.0, n=200, device="cpu"):
    """
    Plota P(Y=1) vs habilidade no tópico k, mantendo os outros tópicos fixos.
    answer_emb: tensor [K] (o vetor do NTM para UMA resposta/item)
    k: índice do tópico interno do CAIMIRA (0..D-1)
    """
    model.eval()
    model = model.to(device)

    # 1) Computa d_y e r_y para o item
    y = answer_emb.to(device).float()          # [1, K]    
    d_raw = model.layer_d(y)                     # [1, D]
            
    # cuidado: d_norm no seu forward usa mean do batch.
    # aqui batch=1 => mean = ele mesmo => d_norm=0, o que não é desejável p/ plot.
    # então vamos usar d_raw direto como "d" para curvas estáveis:
    d = d_raw.squeeze(0)                                    # [D]    

    r_raw = model.layer_r(y).squeeze(0)                     # [D]
    r = torch.softmax(r_raw / model.config.r_temperature, dim=-1)  # [D]

    # 2) pega embedding do usuário
    u = torch.tensor([u_id], device=device, dtype=torch.long)
    s_base = model.layer_s(u).squeeze(0).clone()            # [D]

    # 3) grid de habilidade no tópico k
    x = torch.linspace(x_min, x_max, n, device=device)
    probs = []
    
    #print(f"s: {s_base}")
    #print(f"d: {d}")
    #print(f"r: {r}")
    
    # parte constante do logit (todos os tópicos exceto k)
    const = ((s_base - d) * r).sum() - (s_base[k] - d[k]) * r[k]
    #print(f"const: {const.item():.3f}")
    y = y.squeeze(0)  # [K]
    for xi in x:
        #logit =  const + (xi - d[k]* r[k] ) * y[k] #* r[k] #* r[k] # + const  # logit para este valor de habilidade no tópico k        
        #s_xi = torch.zeros_like(s_base) + xi
        logit = (xi - d[k]) * r[k]
        probs.append(torch.sigmoid(logit).item())
            
    return x.cpu().numpy(), probs, r.cpu().numpy(), d.cpu().numpy()

@torch.no_grad()
def plot_top_topic_curves(model, u_id, answer_emb, top=5, device="cpu"):
    # calcula r e d uma vez
    x, _, r, d = topic_curve_for_item(model, u_id, answer_emb, k=0, device=device)
    top_idx = torch.tensor(r).argsort(descending=True)[:top].tolist()

    plt.figure()
    for k in top_idx:
        xk, pk, _, _ = topic_curve_for_item(model, u_id, answer_emb, k=k, device=device)
        plt.plot(xk, pk, label=f"k={k} (r={r[k]:.3f}, d={d[k]:.2f})")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Habilidade no tópico (s_u[k])")
    plt.ylabel("P(Y=1)")
    plt.title(f"Top-{top} curvas por tópico (mesmo item)")
    plt.legend()
    plt.show()


# ---------- exemplo de uso ----------
# u_id = 0
# answer_emb = batch["answer_emb"][0]  # [K]
# k = 3  # tópico interno
# x, p, r, d = topic_curve_for_item(model, u_id, answer_emb, k, device="cuda")

# plt.figure()
# plt.plot(x, p)
# plt.ylim(-0.02, 1.02)
# plt.xlabel(f"Habilidade no tópico k={k} (s_u[k])")
# plt.ylabel("P(Y=1)")
# plt.title(f"Curva por tópico (k={k})")
# plt.show()



if __name__ == "__main__":
    main(sys.argv[1:])