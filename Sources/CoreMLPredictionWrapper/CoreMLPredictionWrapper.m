#import "CoreMLPredictionWrapper.h"

@implementation CoreMLPredictionWrapper

+ (nullable id<MLFeatureProvider>)predictWithModel:(MLModel *)model
                                             input:(id<MLFeatureProvider>)input
                                             error:(NSError **)error {
    __block id<MLFeatureProvider> result = nil;
    __block NSError *predictionError = nil;

    @autoreleasepool {
        result = [model predictionFromFeatures:input error:&predictionError];
    }

    if (predictionError && error) {
        *error = predictionError;
    }

    return result;
}

+ (nullable NSArray<id<MLFeatureProvider>> *)batchPredictWithModel:(MLModel *)model
                                                            inputs:(NSArray<id<MLFeatureProvider>> *)inputs
                                                     drainInterval:(NSUInteger)drainInterval
                                                             error:(NSError **)error {
    NSMutableArray<id<MLFeatureProvider>> *results = [NSMutableArray arrayWithCapacity:inputs.count];

    for (NSUInteger i = 0; i < inputs.count; i++) {
        @autoreleasepool {
            NSUInteger batchEnd = MIN(i + drainInterval, inputs.count);
            for (NSUInteger j = i; j < batchEnd; j++) {
                NSError *predictionError = nil;
                id<MLFeatureProvider> result = [model predictionFromFeatures:inputs[j] error:&predictionError];
                if (predictionError) {
                    if (error) {
                        *error = predictionError;
                    }
                    return nil;
                }
                [results addObject:result];
            }
            i = batchEnd - 1;  // -1 because the for loop will increment
        }
    }

    return [results copy];
}

@end
