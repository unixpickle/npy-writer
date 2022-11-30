use crate::NumpyWritable;
use std::io::Write;
use std::{fs::File, io::BufWriter, path::Path};
use zip::result::ZipResult;
use zip::{write::FileOptions, ZipWriter};

pub struct NpzWriter {
    writer: ZipWriter<File>,
}

impl NpzWriter {
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<NpzWriter> {
        let file = File::create(path)?;
        let zw = ZipWriter::new(file);
        Ok(NpzWriter { writer: zw })
    }

    pub fn write<T: NumpyWritable>(&mut self, name: &str, t: T) -> std::io::Result<()> {
        let full_name = format!("{}.npy", name);
        self.writer.start_file(&full_name, FileOptions::default())?;
        let mut buffered = BufWriter::new(&mut self.writer);
        t.write_npy(&mut buffered)?;
        buffered.flush()?;
        Ok(())
    }

    pub fn close(&mut self) -> ZipResult<()> {
        self.writer.finish().map(|_| ())
    }
}

impl Drop for NpzWriter {
    fn drop(&mut self) {
        self.close().ok();
    }
}
