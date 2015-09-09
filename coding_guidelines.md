- helper functions should raise exceptions, not call a non-zero exit function so that higher level code can potentially deal with the exception. High level code might use sys.exit(1) as long as we make sure that any printed error messages go to stderr
- (python standard) CamelCase for classes and snake_case for everything else

