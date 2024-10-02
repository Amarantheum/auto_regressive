
mod auto_correlation;
mod yule_walker;

pub struct AutoRegressiveModel {
    coefficients: Vec<f64>,
    noise_variance: f64,
}


impl AutoRegressiveModel {
    pub fn new_with_order(signal: &[f64], order: usize) -> Self {
        assert!(signal.len() > 0);
        assert!(order < signal.len());
        assert!(order > 0);
        
        let result = yule_walker::yule_walker(signal, order);
        Self {
            coefficients: result.coefficients,
            noise_variance: result.noise_variance,
        }
    }
}