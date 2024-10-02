use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

pub fn auto_correlation_fft(signal: &[f64]) -> Vec<f64> {
    let mut planner = FftPlanner::new();

    let fft_size = signal.len() * 2;

    let fft = planner.plan_fft_forward(fft_size);

    let mut buffer = Vec::with_capacity(fft_size);
    for &value in signal {
        buffer.push(Complex { re: value, im: 0.0 });
    }
    for _ in 0..(fft_size - signal.len()) {
        buffer.push(Complex { re: 0.0, im: 0.0 });
    }
    fft.process(&mut buffer);

    let mut power_spectral_density = buffer.iter().map(|value| Complex {re: value.norm_sqr(), im: 0.0}).collect::<Vec<Complex<f64>>>();

    let ifft = planner.plan_fft_inverse(fft_size);
    ifft.process(&mut power_spectral_density);

    let mut auto_correlation = Vec::with_capacity(signal.len());
    for value in power_spectral_density.iter().take(signal.len()) {
        auto_correlation.push(value.re / fft_size as f64);
    }
    auto_correlation
}

#[allow(unused)]
#[deprecated(note = "Use auto_correlation_fft instead")]
pub fn auto_correlation_time_domain(signal: &[f64]) -> Vec<f64> {
    let mut auto_correlation = Vec::with_capacity(signal.len());
    for lag in 0..signal.len() {
        let mut sum = 0.0;
        for i in 0..(signal.len() - lag) {
            sum += signal[i] * signal[i + lag];
        }
        auto_correlation.push(sum);
    }
    auto_correlation
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;
    use rand::Rng;

    #[test]
    fn test_auto_correlation_fft() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let auto_correlation = auto_correlation_fft(&signal);
        let expected = vec![55.0, 40.0, 26.0, 14.0, 5.0];
        
        assert_eq!(auto_correlation.len(), expected.len());
        for (actual, expected) in auto_correlation.iter().zip(expected.iter()) {
            assert_float_eq!(*actual, *expected, abs <= 1e-9);
        }
    }

    #[allow(deprecated)]
    #[test]
    fn test_auto_correlation_time_domain() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let auto_correlation = auto_correlation_time_domain(&signal);
        let expected = vec![55.0, 40.0, 26.0, 14.0, 5.0];
        
        assert_eq!(auto_correlation.len(), expected.len());
        for (actual, expected) in auto_correlation.iter().zip(expected.iter()) {
            assert_float_eq!(*actual, *expected, abs <= 1e-9);
        }
    }

    #[allow(deprecated)]
    #[test]
    fn bench_methods() {
        use std::time::Instant;
        let mut rng = rand::thread_rng();
        let signal: Vec<f64> = (0..1000).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let start = Instant::now();
        let _auto_correlation = auto_correlation_fft(&signal);
        let duration = start.elapsed();
        println!("FFT: {:?}", duration);

        let start = Instant::now();
        let _auto_correlation = auto_correlation_time_domain(&signal);
        let duration = start.elapsed();
        println!("Time domain: {:?}", duration);

        // FFT is already faster at 1000 samples
    }
}