use std::{fmt::Write, ops::Deref};

pub trait NumpyArrayElement {
    const DATA_SIZE: usize;
    const DATA_FORMAT: &'static str;

    fn encode_npy_element<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()>;
}

macro_rules! array_element {
    ($dtype:ty, $data_size:literal, $data_format:literal) => {
        impl NumpyArrayElement for $dtype {
            const DATA_SIZE: usize = $data_size;
            const DATA_FORMAT: &'static str = $data_format;

            fn encode_npy_element<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
                out.write_all(&self.to_le_bytes())
            }
        }
    };
}

array_element!(f32, 4, "<f4");
array_element!(f64, 8, "<f8");
array_element!(u8, 1, "<u1");
array_element!(u16, 2, "<u2");
array_element!(u32, 4, "<u4");
array_element!(u64, 8, "<u8");
array_element!(i8, 1, "<i1");
array_element!(i16, 2, "<i2");
array_element!(i32, 4, "<i4");
array_element!(i64, 8, "<i8");

pub trait NumpyArray: Sized {
    type Elem: NumpyArrayElement;
    type Iter: Iterator<Item = Self::Elem>;

    fn npy_shape(&self) -> Vec<usize>;
    fn npy_elements(self) -> Self::Iter;
}

impl<A: NumpyArrayElement> NumpyArray for A {
    type Elem = A;
    type Iter = <[A; 1] as IntoIterator>::IntoIter;

    fn npy_shape(&self) -> Vec<usize> {
        Vec::new()
    }

    fn npy_elements(self) -> Self::Iter {
        [self].into_iter()
    }
}

impl<A: NumpyArrayElement> NumpyArray for Vec<A> {
    type Elem = A;
    type Iter = <Self as IntoIterator>::IntoIter;

    fn npy_shape(&self) -> Vec<usize> {
        vec![self.len()]
    }

    fn npy_elements(self) -> Self::Iter {
        self.into_iter()
    }
}

pub trait NumpyWriter: Sized {
    fn write_npy<W: std::io::Write>(self, out: &mut W) -> std::io::Result<()>;
}

impl<A: NumpyArray> NumpyWriter for A {
    fn write_npy<W: std::io::Write>(self, out: &mut W) -> std::io::Result<()> {
        out.write_all(&encode_header(
            <Self as NumpyArray>::Elem::DATA_FORMAT,
            &self.npy_shape(),
        ))?;
        for elem in self.npy_elements() {
            elem.encode_npy_element(out)?;
        }
        Ok(())
    }
}

impl NumpyWriter for &str {
    fn write_npy<W: std::io::Write>(self, out: &mut W) -> std::io::Result<()> {
        let chars = self.chars().map(|x| x as u32).collect::<Vec<_>>();
        out.write_all(&encode_header(&format!("<U{}", chars.len()), &[]))?;
        let mut byte_str = Vec::new();
        for ch in chars {
            byte_str.extend(ch.to_le_bytes());
        }
        out.write_all(&byte_str)
    }
}

impl NumpyWriter for &[String] {
    fn write_npy<W: std::io::Write>(self, out: &mut W) -> std::io::Result<()> {
        write_strings_to_npy(self.iter(), out)
    }
}

impl NumpyWriter for &[&str] {
    fn write_npy<W: std::io::Write>(self, out: &mut W) -> std::io::Result<()> {
        write_strings_to_npy(self.iter(), out)
    }
}

pub fn write_strings_to_npy<
    'a,
    S: 'a + AsRef<str>,
    I: 'a + Iterator<Item = S>,
    W: std::io::Write,
>(
    it: I,
    out: &mut W,
) -> std::io::Result<()> {
    let unicode: Vec<Vec<u32>> = it
        .map(|x| x.as_ref().chars().map(|x| x as u32).collect::<Vec<_>>())
        .collect();
    let max_result = unicode.iter().map(|x| x.len()).max();
    if let Some(max_len) = max_result {
        out.write_all(&encode_header(&format!("<U{}", max_len), &[unicode.len()]))?;
        for ustr in unicode {
            let mut byte_str = Vec::new();
            for ch in &ustr {
                byte_str.extend(ch.to_le_bytes());
            }
            for _ in ustr.len()..max_len {
                byte_str.extend([0, 0, 0, 0]);
            }
            out.write_all(&byte_str)?;
        }
        Ok(())
    } else {
        out.write_all(&encode_header("<U1", &[0]))
    }
}

macro_rules! impl_writer_for_deref {
    ($dtype:ty) => {
        impl NumpyWriter for $dtype {
            fn write_npy<W: std::io::Write>(self, out: &mut W) -> std::io::Result<()> {
                self.deref().write_npy(out)
            }
        }
    };
}

impl_writer_for_deref!(String);
impl_writer_for_deref!(&String);
impl_writer_for_deref!(Vec<String>);
impl_writer_for_deref!(&Vec<String>);
impl_writer_for_deref!(Vec<&str>);
impl_writer_for_deref!(&Vec<&str>);

fn encode_header(format: &str, shape: &[usize]) -> Vec<u8> {
    let mut header_data = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': (",
        format
    );
    for (i, n) in shape.iter().enumerate() {
        if i > 0 {
            write!(header_data, " ").unwrap();
        }
        write!(header_data, "{},", n).unwrap();
    }
    write!(header_data, "), }}").unwrap();
    let mut header_bytes = header_data.into_bytes();
    while (11 + header_bytes.len()) % 64 != 0 {
        header_bytes.push(0x20);
    }
    header_bytes.push(0x0a);
    let mut all_bytes = Vec::new();
    all_bytes.push(0x93);
    all_bytes.extend("NUMPY".as_bytes());
    all_bytes.push(1);
    all_bytes.push(0);
    all_bytes.push((header_bytes.len() & 0xff) as u8);
    all_bytes.push(((header_bytes.len() >> 8) & 0xff) as u8);
    all_bytes.extend(header_bytes);
    all_bytes
}

#[cfg(test)]
mod tests {
    use crate::NumpyWriter;

    #[test]
    fn test_encode_strings() {
        let strs = vec!["hello", "test", "123", "longest"];
        let mut buf = Vec::new();
        strs.write_npy(&mut buf).unwrap();
        assert_eq!(buf, include_bytes!("test_data/strings_test_out.npy"));
    }

    #[test]
    fn test_encode_string() {
        let x = "hello".to_owned();
        x.write_npy(&mut std::fs::File::create("foo.npy").unwrap())
            .unwrap();
    }
}
