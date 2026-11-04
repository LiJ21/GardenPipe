use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct EwmaKwargs {
    half_life_ns: f64,
}

#[polars_expr(output_type=Float64)]
fn ewma_unnormalized(inputs: &[Series], kwargs: EwmaKwargs) -> PolarsResult<Series> {
    let input_series = inputs[0].cast(&DataType::Float64)?;
    let time_series = inputs[1].cast(&DataType::Int64)?;
    let input = input_series.f64()?;
    let time = time_series.i64()?;

    polars_ensure!(
        kwargs.half_life_ns.is_finite() && kwargs.half_life_ns > 0.0,
        ComputeError: "half_life_ns must be positive and finite"
    );

    let lambda = std::f64::consts::LN_2 / kwargs.half_life_ns;
    let mut out = Vec::with_capacity(input.len());
    let mut state = 0.0_f64;
    let mut prev_time: Option<i64> = None;

    for i in 0..input.len() {
        let x = input.get(i).unwrap_or(0.0);
        if let Some(t) = time.get(i) {
            if let Some(prev) = prev_time {
                let dt = t.saturating_sub(prev).max(0) as f64;
                state *= (-lambda * dt).exp();
            }
            prev_time = Some(t);
        }
        state += x;
        out.push(state);
    }

    Ok(Series::new(input_series.name().clone(), out))
}

#[polars_expr(output_type=Float32)]
fn ewma_unnormalized_f32(inputs: &[Series], kwargs: EwmaKwargs) -> PolarsResult<Series> {
    let input_series = inputs[0].cast(&DataType::Float32)?;
    let time_series = inputs[1].cast(&DataType::Int64)?;
    let input = input_series.f32()?;
    let time = time_series.i64()?;

    polars_ensure!(
        kwargs.half_life_ns.is_finite() && kwargs.half_life_ns > 0.0,
        ComputeError: "half_life_ns must be positive and finite"
    );

    let lambda = std::f32::consts::LN_2 / kwargs.half_life_ns as f32;
    let mut out = Vec::with_capacity(input.len());
    let mut state = 0.0_f32;
    let mut prev_time: Option<i64> = None;

    for i in 0..input.len() {
        let x = input.get(i).unwrap_or(0.0);
        if let Some(t) = time.get(i) {
            if let Some(prev) = prev_time {
                let dt = t.saturating_sub(prev).max(0) as f32;
                state *= (-lambda * dt).exp();
            }
            prev_time = Some(t);
        }
        state += x;
        out.push(state);
    }

    Ok(Series::new(input_series.name().clone(), out))
}
