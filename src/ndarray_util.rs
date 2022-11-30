use crate::base::{NumpyArray, NumpyArrayElement};
use ndarray::{Array, ArrayBase, Data, Dimension};

impl<E: Data, D: Dimension> NumpyArray for ArrayBase<E, D>
where
    E::Elem: NumpyArrayElement + Clone,
{
    type Elem = E::Elem;
    type Iter = <Array<E::Elem, D> as IntoIterator>::IntoIter;

    fn npy_shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }

    fn npy_elements(self) -> Self::Iter {
        self.into_owned().into_iter()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array3;

    use crate::NumpyWritable;

    #[test]
    fn test_3d_array() {
        let mut arr = Array3::zeros((2, 3, 4));
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    arr[(i, j, k)] = (i as f32) * 0.25 + (j as f32) * 0.5 + (k as f32) * 1.0;
                }
            }
        }
        let mut buf = Vec::new();
        arr.write_npy(&mut buf).unwrap();
        assert_eq!(buf, include_bytes!("test_data/ndarray_test_out.npy"));
    }
}
