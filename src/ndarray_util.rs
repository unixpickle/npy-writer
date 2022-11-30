use crate::{NumpyArray, NumpyArrayElement};
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
