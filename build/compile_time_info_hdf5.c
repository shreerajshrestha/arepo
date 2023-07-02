#include <stdio.h>
#include "arepoconfig.h"
#ifdef HAVE_HDF5
#include <hdf5.h>

hid_t my_H5Acreate(hid_t loc_id, const char *attr_name, hid_t type_id, hid_t space_id, hid_t acpl_id);
hid_t my_H5Screate(H5S_class_t type);
herr_t my_H5Aclose(hid_t attr_id, const char *attr_name);
herr_t my_H5Awrite(hid_t attr_id, hid_t mem_type_id, const void *buf, const char *attr_name);
herr_t my_H5Sclose(hid_t dataspace_id, H5S_class_t type);

herr_t my_H5Tclose(hid_t type_id);

void write_compile_time_options_in_hdf5(hid_t handle)
{
hid_t hdf5_dataspace, hdf5_attribute;
double val;
hid_t atype = H5Tcopy(H5T_C_S1);
H5Tset_size(atype, 1);
my_H5Tclose(atype);
}
#endif
