use nalgebra::{DMatrix, DVector};
use crate::auto_correlation;

#[derive(Debug)]
pub struct YuleWalkerResult {
    pub coefficients: Vec<f64>,
    pub noise_variance: f64,
}

/// Compute the Yule-Walker equations for an AR model of order `p`.
pub fn yule_walker(signal: &[f64], p: usize) -> YuleWalkerResult {
    let mut auto_correlation = auto_correlation::auto_correlation_fft(signal);
    auto_correlation.truncate(p + 1);

    let correlation_matrix = DMatrix::from_fn(p, p, |i, j| {
        // note that can't use abs here because of underflow
        auto_correlation[usize::max(i, j) - usize::min(i, j)]
    });

    let r = DVector::from_iterator(p, auto_correlation.iter().skip(1).cloned());

    let coefficients = correlation_matrix.lu().solve(&r).unwrap();
    let coefficients = coefficients.iter().cloned().collect::<Vec<f64>>();

    let noise_variance = auto_correlation[0] - coefficients.iter().zip(auto_correlation.iter().skip(1)).map(|(a, b)| a * b).sum::<f64>();
    
    YuleWalkerResult {
        coefficients,
        noise_variance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;

    #[test]
    fn test_yule_walker() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = yule_walker(&signal, 2);
        let expected_coefficients = vec![232.0 / 285.0, -34.0 / 285.0];
        let expected_noise_variance = 55.0 - 232.0 / 285.0 * 40.0 + 34.0 / 285.0 * 26.0;

        assert_eq!(result.coefficients.len(), expected_coefficients.len());
        for (actual, expected) in result.coefficients.iter().zip(expected_coefficients.iter()) {
            assert_float_eq!(*actual, *expected, abs <= 1e-9);
        }

        assert_float_eq!(result.noise_variance, expected_noise_variance, abs <= 1e-9);
    }
}