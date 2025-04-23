from enum import Enum, auto
import numpy as np
import arviz
import nutpie
from nutpie.compiled_pyfunc import from_pyfunc

# statistics collector of samples
class Statistic(arviz.data.inference_data.InferenceData):
    def __init__(self, inference_data=None):
        """Initialize with optional InferenceData
        Args:
            inference_data: Existing InferenceData object to wrap
        """
        if inference_data is not None:
            self.__dict__.update(inference_data.__dict__)
        
        self._init_stats_lists()
    
    def _init_stats_lists(self):
        """Initialize statistic category lists"""
        self.basic_stats = [
            'depth', 'maxdepth_reached', 'logp', 'energy',
            'diverging', 'step_size', 'step_size_bar', 'n_steps'
        ]
        self.detailed_stats = [
            'gradient', 'unconstrained_draw',
            'divergence_start', 'divergence_end', 'divergence_momentum'
        ]
    
    def save_to_txt(self, filename="sampling_stats.txt", mode="basic"):
        """Save statistics to text file"""
        if not hasattr(self, 'sample_stats'):
            raise ValueError("No sampling data available")
            
        with open(filename, 'w') as f:
            f.write("Sampling Statistics Report\n")
            f.write("="*30 + "\n\n")
            
            if mode == "basic":
                self._save_basic_stats(f)
            elif mode == "detailed":
                self._save_basic_stats(f)
                self._save_detailed_stats(f)
            else:
                raise ValueError("Invalid mode. Use 'basic' or 'detailed'")
    
    def _save_basic_stats(self, file_obj):
        """Helper method to save basic stats"""
        file_obj.write("\nBasic Statistics:\n")
        file_obj.write("-"*20 + "\n")
        
        stats = self.sample_stats
        for stat_name in self.basic_stats:
            if hasattr(stats, stat_name):
                values = getattr(stats, stat_name).values
                file_obj.write(f"{stat_name}: {values.mean():.4f} (mean)\n")
    
    def _save_detailed_stats(self, file_obj):
        """Helper method to save detailed stats"""
        file_obj.write("\nDetailed Statistics:\n")
        file_obj.write("-"*20 + "\n")
        
        stats = self.sample_stats
        for stat_name in self.detailed_stats:
            if hasattr(stats, stat_name):
                values = getattr(stats, stat_name).values
                file_obj.write(f"\n{stat_name}:\n")
                file_obj.write(str(values) + "\n")
    

class MassMatrixAdaptation(Enum):
    STANDARD = auto()        # standart full matrix
    LOW_RANK = auto()        # low-rank adaptation
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

        fit = nutpie.sample(model, **final_args) # sampling process

        self.statistics = Statistic(fit)
        self.sample = fit.posterior.y

        return self.sample

