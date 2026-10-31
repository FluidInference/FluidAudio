/// Holds a stored property whose type comes from a framework that may be missing at run time
/// (Core AI on macOS 26 and earlier), behind a pointer.
///
/// Every non-generic Swift class is registered with the Objective-C runtime, and
/// `objc_copyClassList` completes each one's metadata, including the layout of its stored
/// properties. For a property of a resilient value type from a weak-linked framework that
/// means calling the type's metadata accessor, which is null when the framework isn't there,
/// so the process crashes even though the class is `@available` and never instantiated.
/// XCTest enumerates classes this way before running any test.
///
/// A generic class is never in that list, and a class reference has a fixed layout, so a
/// property stored as `@CoreAIStorage` keeps the enclosing class's metadata free of the
/// framework's types.
@propertyWrapper
final class CoreAIStorage<Value> {
    var wrappedValue: Value

    init(wrappedValue: Value) {
        self.wrappedValue = wrappedValue
    }
}
