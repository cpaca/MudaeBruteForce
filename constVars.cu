#ifndef MUDAEBRUTEFORCE_CONSTVARS
#define MUDAEBRUTEFORCE_CONSTVARS
// Turns out if I do this and #include this file, it works fine.
// Maximum number of bundles/series that can be activated.
const std::uint32_t MAX_DL = 50;
// Maximum number of free bundles.
// Can be changed whenever, but keep it low or CUDA will demand much more memory than necessary.
const std::uint32_t MAX_FREE_BUNDLES = 5;
// Overlap limit, defined in Mudae
const std::uint32_t OVERLAP_LIMIT = 30000;
// Size of disabledSets.
// Making this const lets me do cool stuff with the Task struct.
const std::uint32_t DISABLED_SETS_SIZE = MAX_DL + MAX_FREE_BUNDLES;
#endif