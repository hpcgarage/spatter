#ifndef UNUSED_H
#define UNUSED_H
// Use this for specifying unused variables in a function declaration
#ifdef __GNUC__
#  define UNUSED(x) UNUSED_ ## x __attribute__((__unused__))
#else
#  define UNUSED(x) UNUSED_ ## x
#endif

// Use this for suppressing warnings when variables are sometimes used
#define UNUSED_VAR(expr) do { (void)(expr); } while (0)
#endif
