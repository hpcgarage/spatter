#ifndef SP_ALLOC_H
#define SP_ALLOC_H
#ifndef SP_MAX_ALLOC
  //20GB
  #define SP_MAX_ALLOC (65ll * 1000 * 1000 * 1000) 
#endif
#define ALIGN_CACHE 64
#define ALIGN_PAGE  4096
void *sp_malloc (size_t size, size_t count, size_t align);
void *sp_calloc (size_t size, size_t count, size_t align);
#endif
