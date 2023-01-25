use anyhow::Result;
use arrow_array::{ArrayRef, Float32Array, Int32Array, RecordBatch};
use clap::Parser;
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use std::{
    collections::HashMap, fs::File, path::PathBuf, str::FromStr, sync::Arc,
};
use tiff::decoder::{DecodingResult, Limits};

#[derive(Parser)]
struct Cli {
    input_path: PathBuf,
    #[arg(short = 'o', long = "output")]
    output_path: Option<PathBuf>,
    #[arg(long = "group")]
    group: Option<f64>,
    #[arg(long = "drop-below", default_value = "1")]
    drop_below: i32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("Loading file");
    let path = PathBuf::from_str("./ShipDensity_Passenger1.tif")?;
    let file = File::open(path)?;
    let mut decoder = tiff::decoder::Decoder::new(file)?.with_limits(Limits::unlimited());
    let (width, height) = decoder.dimensions()?;
    let image = decoder.read_image()?;

    if let DecodingResult::I32(pixels) = image {
        println!("reading pixels");
        let pixel_count = width as u64 * height as u64;
        let progress_bar = ProgressBar::new(pixel_count)
            .with_style(ProgressStyle::with_template("{bar} {percent}% - {elapsed_precise}")?);
        let data = pixels
            .into_iter()
            .progress_with(progress_bar)
            .enumerate()
            .filter(|(_, value)| *value >= cli.drop_below)
            .map(|(idx, value)| (idx % width as usize, idx / width as usize, value))
            .collect::<Vec<_>>();

        println!("converting data");
        let mut data: Vec<(f64, f64, i32)> = data
            .into_iter()
            .map(|(x, y, value)| {
                let lat = lerp(x as f64, (0.0, width as f64), (-180.0, 180.0));
                let lon = lerp(y as f64, (0.0, height as f64), (85.0, -85.0));
                (lat, lon, value)
            })
            .collect();

        if let Some(group) = cli.group {
            println!("grouping data");
            let mut grouped = HashMap::<(i32, i32), (f64, f64, i32)>::new();
            for (lat, lon, value) in data.iter() {
                let lat_index = (lat / group).floor() as i32;
                let lon_index = (lon / group).floor() as i32;
                let grouped_lat = lat_index as f64 * group;
                let grouped_lon = lon_index as f64 * group;
                let entry =
                    grouped
                        .entry((lat_index, lon_index))
                        .or_insert((grouped_lat, grouped_lon, 0));
                entry.2 += value;
            }
            data = grouped.into_values().collect();
        }

        println!("writing parquet");
        let output_path = cli
            .output_path
            .unwrap_or_else(|| cli.input_path.with_extension("parquet"));

        let props = WriterProperties::builder().build();
        let file = File::create(&output_path)?;

        let lat_col = Float32Array::from_iter(data.iter().map(|r| r.0 as f32));
        let lon_col = Float32Array::from_iter(data.iter().map(|r| r.1 as f32));
        let value_col = Int32Array::from_iter(data.iter().map(|r| r.2));

        let batch = RecordBatch::try_from_iter(vec![
            ("lat", Arc::new(lat_col) as ArrayRef),
            ("lon", Arc::new(lon_col) as ArrayRef),
            ("value", Arc::new(value_col) as ArrayRef),
        ])?;

        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        println!("done");
    } else {
        let image_type = match &image {
            DecodingResult::U8(_) => "U8",
            DecodingResult::U16(_) => "U16",
            DecodingResult::U32(_) => "U32",
            DecodingResult::U64(_) => "U64",
            DecodingResult::F32(_) => "F32",
            DecodingResult::F64(_) => "F64",
            DecodingResult::I8(_) => "I8",
            DecodingResult::I16(_) => "I16",
            DecodingResult::I32(_) => "I32",
            DecodingResult::I64(_) => "I64",
        };

        anyhow::bail!("Unexpected image type. Expected I32 but got {}", image_type);
    }

    Ok(())
}

fn lerp(v: f64, domain: (f64, f64), range: (f64, f64)) -> f64 {
    (v - domain.0) / (domain.1 - domain.0) * (range.1 - range.0) + range.0
}

#[cfg(test)]
mod tests {
    use crate::lerp;

    fn assert_approx(actual: f64, expected: f64) {
        assert!((expected - actual).abs() < 0.001, "{} should be approximately to {}", actual, expected);
    }

    #[test]
    fn test_lerp() {
        assert_approx(lerp(0.5, (0.0, 1.0), (-10.0, 10.0)), 0.0);
        assert_approx(lerp(0.2, (0.0, 2.0), (-3.0, 5.0)), -2.2);
        assert_approx(lerp(0.75, (-1.0, 1.0), (-10.0, 10.0)), 7.5);
    }

    #[test]
    fn test_lerp_reverse() {
        assert_approx(lerp(0.1, (0.0, 1.0), (100.0, 0.0)), 90.0);
        assert_approx(lerp(0.9, (0.0, 1.0), (100.0, 0.0)), 10.0);
    }
}
