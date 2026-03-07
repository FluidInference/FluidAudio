//
//  lapjv.cpp
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/17/25.
//

#include <memory>
#include <cmath>
#include "include/RLAP.hpp"
#include "CrouseSAP.h"


long solveRectangularLinearAssignment(long numrows, long numcols,
                                      const long* cost, bool maximize,
                                      long* rowsol, long* colsol) noexcept {
    return solveRectangularLinearAssignment<long, long>(numrows, numcols,
                                                        cost, maximize,
                                                        rowsol, colsol);
}

long solveRectangularLinearAssignment(long numrows, long numcols,
                                      const float* cost, bool maximize,
                                      long* rowsol, long* colsol) noexcept {
    return solveRectangularLinearAssignment<long, float>(numrows, numcols,
                                                         cost, maximize,
                                                         rowsol, colsol);
}

long solveRectangularLinearAssignment(long numrows, long numcols,
                                      const double* cost, bool maximize,
                                      long* rowsol, long* colsol) noexcept {
    return solveRectangularLinearAssignment<long, double>(numrows, numcols,
                                                          cost, maximize,
                                                          rowsol, colsol);
}
