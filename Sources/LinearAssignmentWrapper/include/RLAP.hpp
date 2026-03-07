#pragma once
#include <cstdint>
#include "RLAP.hpp"

long solveRectangularLinearAssignment(long numrows, long numcols,
                                      const long* cost, bool maximize,
                                      long* rowsol, long* colsol) noexcept;

long solveRectangularLinearAssignment(long numrows, long numcols,
                                      const float* cost, bool maximize,
                                      long* rowsol, long* colsol) noexcept;

long solveRectangularLinearAssignment(long numrows, long numcols,
                                      const double* cost, bool maximize,
                                      long* rowsol, long* colsol) noexcept;

