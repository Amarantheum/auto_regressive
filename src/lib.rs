mod auto_correlation;
mod yule_walker;

#[derive(Debug)]
pub struct AutoRegressiveModel {
    coefficients: Vec<f64>,
    noise_variance: f64,
}


impl AutoRegressiveModel {
    pub fn new_with_order(signal: &[f64], order: usize) -> Self {
        assert!(signal.len() > 0);
        assert!(order < signal.len());
        assert!(order > 0);

        let auto_correlation = auto_correlation::auto_correlation_fft(signal);
        let result = yule_walker::yule_walker_from_auto_correlation(&auto_correlation[0..order + 1]);
        Self {
            coefficients: result.coefficients,
            noise_variance: result.noise_variance,
        }
    }

    pub fn coefficients(&self) -> &Vec<f64> {
        &self.coefficients
    }

    pub fn noise_variance(&self) -> f64 {
        self.noise_variance
    }
}

use num_complex::Complex;

impl AutoRegressiveModel {
    pub fn get_poles(&self) -> Vec<Complex<f64>> {
        let mut coefs = Vec::with_capacity(self.coefficients.len() + 1);
        // the polynomial is 1 - a_1 * z^-1 - a_2 * z^-2 - ... - a_p * z^-p
        coefs.push(1.0);
        for coef in self.coefficients.iter() {
            coefs.push(-coef);
        }
        let poly = polynomen::Poly::new_from_coeffs(&coefs[..]);
        poly.complex_roots()
            .iter()
            .map(|root| num_complex::Complex { re: root.0, im: root.1 }.inv() ) // invert the roots since they are in terms of z^(-1)
            .collect()
    }
}

#[allow(unused)]
#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use float_eq::assert_float_eq;
    use rand::Rng;
    use rand_distr::{Normal, Distribution};
    use rustfft::{FftPlanner, num_complex::Complex};
    use plotters::prelude::*;

    fn plot_series(series: &[f64]) {
        let max = series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = series.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);

        let mut plot = BitMapBackend::new("test_auto_regressive_model_known_input.png", (800, 600)).into_drawing_area();
        plot.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&plot)
            .caption("Power Spectral Density", ("sans-serif", 20).into_font())
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0..series.len(), min..max).unwrap();
        chart.configure_mesh().draw().unwrap();
        chart.draw_series(LineSeries::new(series.iter().enumerate().map(|v| (v.0, *v.1)), BLACK)).unwrap();
    }

    #[test]
    fn test_auto_regressive_model() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model = AutoRegressiveModel::new_with_order(&signal, 2);
        let expected_coefficients = vec![232.0 / 285.0, -34.0 / 285.0];
        let expected_noise_variance = 55.0 - 232.0 / 285.0 * 40.0 + 34.0 / 285.0 * 26.0;

        assert_eq!(model.coefficients().len(), expected_coefficients.len());
        for (actual, expected) in model.coefficients().iter().zip(expected_coefficients.iter()) {
            assert_float_eq!(*actual, *expected, abs <= 1e-9);
        }

        assert_float_eq!(model.noise_variance, expected_noise_variance, abs <= 1e-9);
    }

    #[test]
    fn test_auto_regressive_model_poles() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model = AutoRegressiveModel::new_with_order(&signal, 2);
        let poles = model.get_poles();
        let a = 34.0/285.0;
        let b = -232.0/285.0;
        let c = 1.0;
        let expected_poles = vec![
            Complex::new((2.0 * a) / (-b - (b * b - 4_f64 * a * c).sqrt()), 0.0),
            Complex::new((2.0 * a) / (-b + (b * b - 4_f64 * a * c).sqrt()), 0.0),
        ];

        println!("{:?}", poles);
        assert_eq!(poles.len(), expected_poles.len());
        for (actual, expected) in poles.iter().zip(expected_poles.iter()) {
            assert_float_eq!(actual.re, expected.re, abs <= 1e-9);
            assert_float_eq!(actual.im, expected.im, abs <= 1e-9);
        }
    }

    #[test]
    fn test_auto_regressive_model_known_input() {
        let size = 100000;
        let p = 2;
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.5).unwrap();
        let signal: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
        let a1 = 0.5;
        let a2 = -0.3;
        let mut ar_signal = vec![0.0; signal.len()];
        ar_signal[0] = signal[0];
        ar_signal[1] = signal[1];
        for i in p..signal.len() {
            ar_signal[i] = a1 * ar_signal[i - 1] + a2 * ar_signal[i - 2] + signal[i];
        }

        let model = AutoRegressiveModel::new_with_order(&ar_signal, p);
        println!("{:?}", model);
        assert_float_eq!(model.coefficients()[0], a1, abs <= 1e-2);
        assert_float_eq!(model.coefficients()[1], a2, abs <= 1e-2);

        // let mut fftplanner = FftPlanner::new();
        // let fft = fftplanner.plan_fft_forward(size);
        // let mut buffer = ar_signal.iter().map(|&value| Complex { re: value, im: 0.0 }).collect::<Vec<Complex<f64>>>();
        // fft.process(&mut buffer);
        // let power_spectral_density = buffer.iter().map(|value| value.norm_sqr()).collect::<Vec<f64>>();

        //plot_series(&power_spectral_density);
    }

    #[test]
    fn test_auto_regressive_model_known_input_2() {
        let size = 100000;
        let p = 2;
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.5).unwrap();
        let signal: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
        let a1 = 0.999 * 2_f64.sqrt();
        let a2 = -0.999 * 0.999;
        let mut ar_signal = vec![0.0; signal.len()];
        ar_signal[0] = signal[0];
        ar_signal[1] = signal[1];
        for i in p..signal.len() {
            ar_signal[i] = a1 * ar_signal[i - 1] + a2 * ar_signal[i - 2] + signal[i];
        }

        let model = AutoRegressiveModel::new_with_order(&ar_signal, p);
        //println!("{:?}", model);
        assert_float_eq!(model.coefficients()[0], a1, abs <= 1e-2);
        assert_float_eq!(model.coefficients()[1], a2, abs <= 1e-2);

        //println!("{:?}", model.get_poles());
        let poles = model.get_poles();
        let expected_poles = vec![
            Complex::from_polar(0.999, PI / 4.0),
            Complex::from_polar(0.999, -PI / 4.0),
        ];

        assert_eq!(poles.len(), expected_poles.len());
        for (actual, expected) in poles.iter().zip(expected_poles.iter()) {
            assert_float_eq!(actual.re, expected.re, abs <= 1e-2);
            assert_float_eq!(actual.im, expected.im, abs <= 1e-2);
        }

        // let mut fftplanner = FftPlanner::new();
        // let fft = fftplanner.plan_fft_forward(size);
        // let mut buffer = ar_signal.iter().map(|&value| Complex { re: value, im: 0.0 }).collect::<Vec<Complex<f64>>>();
        // fft.process(&mut buffer);
        // let power_spectral_density = buffer.iter().map(|value| value.norm_sqr()).collect::<Vec<f64>>();

        //plot_series(&power_spectral_density);
    }
}