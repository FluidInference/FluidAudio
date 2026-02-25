#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Wraps CoreML prediction calls in ObjC @autoreleasepool to properly release
/// E5/ANE IOSurface buffers that leak in pure Swift autoreleasepool.
@interface CoreMLPredictionWrapper : NSObject

/// Run a single prediction wrapped in @autoreleasepool.
+ (nullable id<MLFeatureProvider>)predictWithModel:(MLModel *)model
                                             input:(id<MLFeatureProvider>)input
                                             error:(NSError **)error;

/// Run multiple predictions, draining autoreleasepool every `drainInterval` predictions.
/// Returns an array of MLFeatureProvider results. On error, returns nil.
+ (nullable NSArray<id<MLFeatureProvider>> *)batchPredictWithModel:(MLModel *)model
                                                            inputs:(NSArray<id<MLFeatureProvider>> *)inputs
                                                     drainInterval:(NSUInteger)drainInterval
                                                             error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
