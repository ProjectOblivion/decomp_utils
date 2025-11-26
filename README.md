# sotn_utils
A collection of python code to assist in decompilation Castlevania: Symphony of the Night.

## Requirements
* Python 3.9 or higher - Written and tested using Python 3.12, there may be unknown incompatibilities with versions earlier than 3.12
* [splat](https://github.com/ethteck/splat) - Tested against 0.27.3
* [Levenshtein](https://github.com/rapidfuzz/Levenshtein) - Tested against 0.27.1

## Limitations
* This project should be considered to be in a pre-alpha state.  As such, doc strings, type hints, and other non-functional elements are sporadic at best.

## Known Issues
* RODATA in wrong segment when it falls before it's function and that function is on a segment boundary

## Things to do, in no particular order or timeline
* Adding "polish" like proper docstrings, type hints, pep8 compliance, etc.
* Implementing and eliminating "todo" comments
* Improving logging and logger usage
* Splitting sotn specific code from code useful for any decomp
* Adding tests

## Special thanks
* [sotn-decomp](https://github.com/Xeeynamo/sotn-decomp) for providing the foundation for this project.  This wouldn't have been possible without the hard work done by @Xeeynamo and others.:
* [decomp.me](https://github.com/decompme/decomp.me/) by @ethteck, @nanaian and @mkst as a collaborative decompilation site to share and contribute to work-in-progress decompiled functions.
* [splat](https://github.com/ethteck/splat) from @ethteck used to disassemble code and extract data with a symbol map.
* Anybody or anything else that has contributed to anything in this project.  If I neglected to give due credit, it is entirely unintentional.