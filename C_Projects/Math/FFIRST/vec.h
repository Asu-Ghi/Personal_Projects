#ifndef VEC_H
#define VEC_H

// calculate the distance squared between dim dimensional vectors u and v
double vec_dist_sq (double* u, double* v, int dim);

// read len vectors in dim dimensional space from stdin into data array
void vec_read_dataset (double* data, int len, int dim);

#endif
