# decomp_utils
A collection of python code to assist in decompilation efforts directly or by extending fantastic tools like [splat](https://github.com/ethteck/splat).

## Limitations
Known limitations:

* It is currently heavily biased toward [sotn-decomp](https://github.com/Xeeynamo/sotn-decomp) and will require a lot of adjustments to be useful for any other project.
* This project should be considered to be in a pre-alpha state.  As such, doc strings, type hints, and other non-functional elements are sporadic at best.

## Project tasks, in no particular order or timeline
* Adding "polish" like proper docstrings, type hints, pep8 compliance, etc.
* Implementing and eliminating "todo" comments
* Improving logging and logger usage
* Splitting sotn specific code from code useful for any decomp
* Adding tests

## Future tasks, to be accomplished eventually
* Become a project that offers usefulness to any decomp project

## Special thanks

* [sotn-decomp](https://github.com/Xeeynamo/sotn-decomp) for providing the foundation for this project.  This wouldn't have been possible without the hard work done by @Xeeynamo and others.:
* [decomp.me](https://github.com/decompme/decomp.me/) by @ethteck, @nanaian and @mkst as a collaborative decompilation site to share and contribute to work-in-progress decompiled functions.
* [splat](https://github.com/ethteck/splat) from @ethteck used to disassemble code and extract data with a symbol map.
* Anybody or anything else that has contributed to anything in this project.  If I neglected to give due credit, it is entirely unintentional.