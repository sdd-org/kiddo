use aligned_vec::{AVec, CACHELINE_ALIGN};
use rkyv::{
    ser::{ScratchSpace, Serializer},
    vec::{ArchivedVec, VecResolver},
    with::{ArchiveWith, DeserializeWith, SerializeWith},
    Archive, Deserialize, Fallible, Serialize,
};
use std::marker::PhantomData;

pub(crate) struct EncodeAVec<T> {
    _p: PhantomData<T>,
}

impl<T: Archive> ArchiveWith<AVec<T>> for EncodeAVec<T> {
    type Archived = ArchivedVec<T::Archived>;
    type Resolver = VecResolver;

    unsafe fn resolve_with(
        field: &AVec<T>,
        pos: usize,
        resolver: Self::Resolver,
        out: *mut Self::Archived,
    ) {
        ArchivedVec::resolve_from_slice(field.as_slice(), pos, resolver, out);
    }
}

impl<T, S> SerializeWith<AVec<T>, S> for EncodeAVec<T>
where
    T: Serialize<S>,
    S: ScratchSpace + Serializer + ?Sized,
{
    fn serialize_with(field: &AVec<T>, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        ArchivedVec::serialize_from_slice(field.as_slice(), serializer)
    }
}

impl<T, D> DeserializeWith<ArchivedVec<T::Archived>, AVec<T>, D> for EncodeAVec<T>
where
    T: Archive,
    T::Archived: Deserialize<T, D>,
    D: Fallible + ?Sized,
{
    fn deserialize_with(
        field: &ArchivedVec<T::Archived>,
        deserializer: &mut D,
    ) -> Result<AVec<T>, D::Error> {
        let mut result = AVec::with_capacity(CACHELINE_ALIGN, field.len());

        for item in field.iter() {
            result.push(item.deserialize(deserializer)?);
        }

        Ok(result)
    }
}
