from delta.models.ntm import print_weights
from types import SimpleNamespace
import lightning as L

class BetaCallBack(L.Callback):
    """Lightning callback to print the beta parameter of the NTM model at the end of each epoch."""
    
    def on_train_epoch_end(self, trainer, pl_module):
        if not hasattr(pl_module, "delta_model"):
            return
        
        model = pl_module.delta_model
        if not hasattr(model, "eta_bn_prop"):
            return
        
        beta = model.eta_bn_prop
        if beta > 0:
                beta -= 1.0 / float(0.75 * trainer.max_epochs)
                if beta < 0:
                    beta = 0.0
        model.eta_bn_prop = beta                    
        print(f"Epoch {pl_module.current_epoch}: eta_bn_prop = {beta}")


class PrintNTMTopics(L.Callback):
    """Lightning callback to print NTM topics (uses NTMModel helper functions).

    Usage: pass an iterable of vocabulary tokens (length == vocab_size) and
    attach the callback to the Trainer via `callbacks=[PrintNTMTopics(vocab)]`.
    """

    def __init__(self, vocab, n_pos: int = 8, n_neg: int = 5, sparsity_threshold: float = 1e-5, every_n_epochs: int = 1, print_bg: bool = True):
        self.vocab = vocab
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.sparsity_threshold = sparsity_threshold
        self.every_n_epochs = max(1, every_n_epochs)
        self.print_bg = print_bg

    def _print_topics_from_model(self, model):
        # Delegate printing to the common print_weights helper.
        try:
            options = SimpleNamespace(
                no_bg=not self.print_bg,
                n_topics=getattr(getattr(model, 'config', None), 'n_topics', None),
                output_dir=None,
                interactions=False,
            )
            print_weights(options, model, self.vocab, topic_covar_names=getattr(model.config, 'topic_covar_names', None))
        except Exception as e:
            print("PrintNTMTopics: cannot print topic weights from model:", e)
            return

    def on_train_epoch_end(self, trainer, pl_module):
        # print only every N epochs
        if (int(pl_module.current_epoch) + 1) % self.every_n_epochs != 0:
            return

        if not hasattr(pl_module, "delta_model"):
            return

        model = pl_module.delta_model
        self._print_topics_from_model(model)


