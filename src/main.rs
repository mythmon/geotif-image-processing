#[allow(unused_imports)]
use anyhow::{anyhow, bail, Result};
use arrow_array::{ArrayRef, Float32Array, RecordBatch};
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle, ProgressIterator};
use parquet::arrow::ArrowWriter;
use std::{
    collections::HashMap,
    fs::File,
    io::{Cursor, Read},
    path::{Path, PathBuf},
    sync::Arc,
};
use tiff::decoder::{DecodingResult, Limits};
use zip::ZipArchive;

#[derive(Parser)]
struct Cli {
    input_path: Vec<PathBuf>,
    #[arg(long = "group")]
    group: Option<f64>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let multi_bar = MultiProgress::new();
    cli.input_path
        .into_iter()
        .map(|input_path| process_one(multi_bar.clone(), input_path, cli.group))
        .collect::<Result<Vec<_>>>()?;
    Ok(())
}

fn process_one(multi_bar: MultiProgress, input_path: PathBuf, group: Option<f64>) -> Result<()> {
    let bar = multi_bar.add(ProgressBar::new_spinner());
    bar.set_style(ProgressStyle::with_template("{prefix:<30} {msg}")?);
    bar.set_prefix(input_path.to_string_lossy().to_string());
    bar.set_message("reading file");
    let tif_contents = load_tif_contents(input_path.as_path())?;

    bar.set_message("decoding tif");
    let mut decoder =
        tiff::decoder::Decoder::new(Cursor::new(tif_contents))?.with_limits(Limits::unlimited());
    let (width, height) = decoder.dimensions()?;
    let image = decoder.read_image()?;

    if let DecodingResult::I32(pixels) = image {
        bar.set_message("processing image");
        bar.set_length(width as u64 * height as u64);
        bar.set_style(ProgressStyle::with_template(
            "{prefix:<30} {msg} {percent}% {elapsed_precise} {bar_wide}",
        )?);

        let mut data: Vec<(f64, f64, f64)> = pixels
            .into_iter()
            .progress_with(bar.clone())
            .enumerate()
            .filter(|(_, value)| *value > 0)
            .map(|(idx, value)| (idx % width as usize, idx / width as usize, value))
            .map(|(x, y, value)| {
                let lon = lerp(x as f64, (0.0, width as f64), (-180.0, 180.0));
                let lat = lerp(y as f64, (0.0, height as f64), (85.0, -85.0));
                (lon, lat, value as f64)
            })
            .collect();

        if let Some(group) = group {
            let mut grouped = HashMap::<(i32, i32), (f64, f64, f64)>::new();
            for (lon, lat, value) in data.iter() {
                let lon_index = (lon / group).floor() as i32;
                let lat_index = (lat / group).floor() as i32;
                let grouped_lon = lon_index as f64 * group;
                let grouped_lat = lat_index as f64 * group;
                let entry = grouped.entry((lon_index, lat_index)).or_insert((
                    grouped_lon,
                    grouped_lat,
                    0.0,
                ));
                let scaled = *value * lat.to_radians().cos();
                entry.2 += scaled;
            }
            data = grouped.into_values().collect();
        }

        let lon_col = Float32Array::from_iter(data.iter().map(|r| r.1 as f32));
        let lat_col = Float32Array::from_iter(data.iter().map(|r| r.0 as f32));
        let value_col = Float32Array::from_iter(data.iter().map(|r| r.2 as f32));

        let batch = RecordBatch::try_from_iter(vec![
            ("lon", Arc::new(lon_col) as ArrayRef),
            ("lat", Arc::new(lat_col) as ArrayRef),
            ("value", Arc::new(value_col) as ArrayRef),
        ])?;

        let output_file = File::create(&input_path.with_extension("parquet"))?;
        let mut writer = ArrowWriter::try_new(output_file, batch.schema(), None)?;
        writer.write(&batch)?;
        writer.close()?;
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

    bar.finish_with_message("done");
    Ok(())
}

fn load_tif_contents(path: &Path) -> Result<Vec<u8>> {
    let mut tif_contents: Vec<u8> = vec![];
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) if ext == "zip" => {
            let zip_file = File::open(path)?;
            let mut archive = ZipArchive::new(zip_file)?;
            let tif_names = archive
                .file_names()
                .filter(|n| n.ends_with(".tif") || n.ends_with(".tiff"))
                .map(|n| n.to_string())
                .collect::<Vec<_>>();
            match &tif_names[..] {
                [] => bail!("No tif files found archive"),
                [tif_name] => archive.by_name(tif_name)?.read_to_end(&mut tif_contents)?,
                _ => bail!("Multiple tif files found in archive"),
            }
        }
        Some(ext) if ext == "tif" => File::open(path)?.read_to_end(&mut tif_contents)?,
        Some(ext) => bail!("Unexpected file extension {}", ext),
        None => bail!("No file extension on {}", path.to_string_lossy()),
    };
    Ok(tif_contents)
}

fn lerp(v: f64, domain: (f64, f64), range: (f64, f64)) -> f64 {
    (v - domain.0) / (domain.1 - domain.0) * (range.1 - range.0) + range.0
}

#[cfg(test)]
mod tests {
    use crate::lerp;

    fn assert_approx(actual: f64, expected: f64) {
        assert!(
            (expected - actual).abs() < 0.001,
            "{} should be approximately to {}",
            actual,
            expected
        );
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
