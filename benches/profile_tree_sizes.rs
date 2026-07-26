pub fn from_env(default_sizes: &[usize]) -> Vec<usize> {
    assert!(
        !default_sizes.is_empty(),
        "default tree sizes must not be empty"
    );

    let default_min = default_sizes.iter().copied().min().unwrap().ilog2();
    let default_max = default_sizes.iter().copied().max().unwrap().ilog2();
    let min = read_log2("KIDDO_PROFILE_MIN_LOG2_POINTS").unwrap_or(default_min);
    let max = read_log2("KIDDO_PROFILE_MAX_LOG2_POINTS").unwrap_or(default_max);

    assert!(
        min <= max,
        "KIDDO_PROFILE_MIN_LOG2_POINTS must not exceed KIDDO_PROFILE_MAX_LOG2_POINTS"
    );
    assert!(
        max <= 31,
        "KIDDO_PROFILE_MAX_LOG2_POINTS must fit u32 item indices"
    );

    if std::env::var_os("KIDDO_PROFILE_MIN_LOG2_POINTS").is_none()
        && std::env::var_os("KIDDO_PROFILE_MAX_LOG2_POINTS").is_none()
    {
        return default_sizes.to_vec();
    }

    (min..=max).map(|log2| 1usize << log2).collect()
}

fn read_log2(var: &str) -> Option<u32> {
    std::env::var(var).ok().map(|value| {
        value
            .parse::<u32>()
            .unwrap_or_else(|_| panic!("{var} must be a non-negative integer"))
    })
}
