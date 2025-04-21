from enum import Enum, auto
import numpy as np
import nutpie
from nutpie.compiled_pyfunc import from_pyfunc

# statistics collector of samples
class Statistic:
    def __init__(self, trace):
        self.trace = trace

        self.basic_stats = [
            'depth',              # Tree depth for current draw
            'maxdepth_reached',   # Whether max tree depth was hit
            'logp',               # Log probability of current position
            'energy',             # Hamiltonian energy
            'diverging',          # Whether the transition diverged
            'step_size',          # Current step size
            'step_size_bar',      # Current estimate of an ideal step size
            'n_steps'             # Number of leapfrog steps
        ]

        self.detailed_stats = [
            'gradient',              # Gradient at current position
            'unconstrained_draw',    # Parameters in unconstrained space
            'divergence_start',      # Position where divergence started
            'divergence_end',        # Position where divergence ended
            'divergence_momentum',   # Momentum at divergence
        ]
    

class MassMatrixAdaptation(Enum):
    STANDARD = auto()        # standart full matrix
    LOW_RANK = auto()        # Low-rank adaptation
    GRADIENT_BASED = auto()  # gradient adaptation

# class for creating samples from a non-analytical distribution
class NutsSampler:
    def __init__(
            self, 
            log_func, 
            dim, 
            cores=None, 
            initial_guess=None, 
            bounds=None
        ):
        
        self.log_func = log_func            # function that return energy and gradient in point
        self.initial_guess = initial_guess  # initial position in coord space
        self.bounds = bounds                # sampler bounds
        self.dim = dim                      # dimension of coord space
        self.sample = None
    
    @staticmethod
    def make_expand_func(*unused):
        def expand(x, **unused):
            return {"y": x}
        return expand
    
    def make_logp_func(self):
        def logp(x : np.ndarray
                , **unused):
            return self.log_func(x)
        return logp
    
    def create_sample(
        self,
        *,
        # the main sampler settings
        draws: int = 100,
        tune: int = 50,
        chains: int = 1,
        target_accept: float = 0.8,
        maxdepth: int = 100,
        max_energy_error: float = 1000.0,
        
        # type of mass matrix adaptation
        mass_matrix_mode: MassMatrixAdaptation = MassMatrixAdaptation.STANDARD,
        
        # statistics settings
        save_warmup: bool = True,
        store_gradient: bool = True,
        store_divergences: bool = True,
        store_unconstrained: bool = True,
        
        **kwargs
    ):

        # base settings
        final_args = {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "target_accept": target_accept,
            "maxdepth": maxdepth,
            "max_energy_error": max_energy_error,
            "save_warmup": save_warmup,
            "store_gradient": store_gradient,
            "store_divergences": store_divergences,
            "store_unconstrained": store_unconstrained,
            **kwargs
        }
        
        # Setting the parameters of the mass matrix depending on the mode
        if mass_matrix_mode == MassMatrixAdaptation.LOW_RANK:
            final_args.update({
                "low_rank_modified_mass_matrix": True,
                "store_mass_matrix": False,
            })
        elif mass_matrix_mode == MassMatrixAdaptation.GRADIENT_BASED:
            final_args.update({
                "use_grad_based_mass_matrix": True,
                "store_mass_matrix": True
            })
        else:  # STANDARD
            final_args.update({
                "use_grad_based_mass_matrix": False,
                "store_mass_matrix": True
            })
        
        model = from_pyfunc(
            self.dim, 
            self.make_logp_func, 
            NutsSampler.make_expand_func, 
            [np.float64], [(self.dim,)], ["y"],
        )

        fit = nutpie.sample(model, **final_args)
        self.sample = fit.posterior.y
        self.statistics = Statistic(fit)
        return self.sample

