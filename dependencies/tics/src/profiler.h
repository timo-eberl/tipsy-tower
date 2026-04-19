#ifndef PROFILER_H
#define PROFILER_H

// Feature macros MUST be defined before any #include
#if defined(__linux__) || defined(__APPLE__) || defined(__unix__)
// Only define it if its not already defined. It is unlikely that it's lower than 1993
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
#endif

#include <stdint.h>
#include <stdio.h>

#ifdef TICS_ENABLE_PROFILER

// --- Cross-platform high precision timing ---

// Linux / POSIX
#if defined(__linux__) || defined(__APPLE__) || defined(__unix__)
#include <time.h>
#include <unistd.h>
#endif

// Windows
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

// MacOS specific (for older versions fallback)
#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

// Returns a timestamp in nanoseconds (platform independent)
static long long time_ns() {
	long long ns = 0;

#if defined(_WIN32) || defined(_WIN64)
	// WINDOWS: Use QueryPerformanceCounter (High Resolution)
	static LARGE_INTEGER frequency;
	static int initialized = 0;
	if (!initialized) {
		QueryPerformanceFrequency(&frequency);
		initialized = 1;
	}

	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);

	// Convert to nanoseconds:  (ticks * 1,000,000,000) / frequency
	// We assume frequency is non-zero (always true on modern Windows)
	ns = (long long)((now.QuadPart * 1000000000LL) / frequency.QuadPart);

#elif defined(__APPLE__)
	// MACOS: Use mach_absolute_time for highest precision on older & newer Macs
	static mach_timebase_info_data_t timebase;
	static int initialized = 0;
	if (!initialized) {
		mach_timebase_info(&timebase);
		initialized = 1;
	}

	uint64_t now = mach_absolute_time();
	ns = (long long)((now * timebase.numer) / timebase.denom);

#elif defined(CLOCK_MONOTONIC)
	// LINUX / POSIX: Use clock_gettime
	struct timespec ts;
	if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
		ns = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
	}
#else
	// FALLBACK (Low precision, but standard C)
	// If we can't find anything better, rely on standard C time()
	ns = (long long)time(NULL) * 1000000000LL;
#endif

	return ns;
}

// --- Profiler ---

// Linked list entry
typedef struct ProfileTimer {
	const char* name;
	uint64_t elapsed_ns;
	uint32_t call_count;
	int depth;
	struct ProfileTimer* next;
	int registered;
} ProfileTimer;

static ProfileTimer* prof_root = NULL;
static ProfileTimer* prof_tail = NULL;
static int prof_depth = 0;

// Generates unique variable name
#define P_CONCAT(a, b) a##b
#define P_VAR(name, line) P_CONCAT(name, line)

#define PROFILE(NAME)                                                                              \
	static ProfileTimer P_VAR(_pt_, __LINE__) = {NAME, 0, 0, 0, NULL, 0};                          \
	/* Registration (Runs once) - FIFO order */                                                    \
	if (!P_VAR(_pt_, __LINE__).registered) {                                                       \
		if (!prof_root) prof_root = &P_VAR(_pt_, __LINE__);                                        \
		else prof_tail->next = &P_VAR(_pt_, __LINE__);                                             \
		prof_tail = &P_VAR(_pt_, __LINE__);                                                        \
		P_VAR(_pt_, __LINE__).registered = 1;                                                      \
	}                                                                                              \
	/* The Scope Loop */                                                                           \
	for (uint64_t _start = (prof_depth++, time_ns()), _once = 1; _once;                            \
		 P_VAR(_pt_, __LINE__).elapsed_ns += (time_ns() - _start),                                 \
				  P_VAR(_pt_, __LINE__).call_count++,                                              \
				  P_VAR(_pt_, __LINE__).depth = prof_depth, /* capture depth */                    \
		 prof_depth--, _once = 0)

static void profile_print() {
	ProfileTimer* t = prof_root;
	fprintf(stderr, "--- Profiler Stats ---\n");
	while (t) {
		if (t->call_count > 0) {
			double avg_ms = (double)t->elapsed_ns / (t->call_count * 1000000.0);

			// indentation for nicely formatted output
			int indent = (t->depth - 1) * 2;
			int width = 30 - indent; // Adjust '30' to be wider than your longest name
			if (width < 0) width = 0;

			// %*s prints indentation
			// %-*s prints name padded to the calculated width
			fprintf(stderr, "%*s%-*s: %8.4f ms (Avg over %d)\n", indent, "", width, t->name, avg_ms,
				   t->call_count);
		}
		t = t->next;
	}
	fprintf(stderr, "----------------------\n");
}

static void profile_reset() {
	ProfileTimer* t = prof_root;
	while (t) {
		t->elapsed_ns = 0;
		t->call_count = 0;
		t = t->next;
	}
}

#else // TICS_ENABLE_PROFILER is not defined

// No-op definitions
#define PROFILE(NAME)
static inline long long time_ns() {
	return 0;
}
static inline void profile_print() {}
static inline void profile_reset() {}

#endif // TICS_ENABLE_PROFILER

#endif // PROFILER_H
