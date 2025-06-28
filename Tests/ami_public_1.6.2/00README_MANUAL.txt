AMI Manual Annotations release 1.7
Date: 16th June 2014
Built by: Jonathan Kilgour
Contact: amicorpus@amiproject.org

Please read LICENCE.txt before using this data.

Please quote the release number in any correspondence.

The annotation data is in a format ready to be used directly by
NXT. Download and further information here:
 http://www.ltg.ed.ac.uk/NITE/
This data requires NXT 1.4.1 or later.

To use this data with AMI media files, make sure the signals you have
downloaded from http://corpus.amiproject.org/ are in a directory
called 'signals' under this directory.

------------------------

Changes in public release 1.7 from 1.6 

Only one change: transcription files for non-scenario meetings updated
to include more accurate and complete timings so that scripts to
extract timing information do not return NaN (not a number) results.


------------------------

Changes in public release 1.6 from 1.5 

For full list of annotations in this release: see MANIFEST_MANUAL.txt

NEW DATA

 * You-usage annotations for 17 meetings contributed by Matthew Purver
   of Queen Mary University of London; created by the CALO Project
   CSLI Stanford team. Transform to NXT: Rieks op den Akker,
   University of Twente

KNOWN ISSUES

The meetings IS1002a or IS1005d were dropped completely from the
corpus because of serious problems with the audio recrdings.

A small number of words are not assigned timings from the
forced-alignment process, causing timing propagation to a small number
of segments to fail. Timings are known to be incomplete / incorrect
for meetings TS3009c (channel 3 only); EN2002a,c; EN2003a. Please
report any other timing issues to jonathan at inf.ed.ac.uk.

There are 28 dialogue-acts in the corpus that are not associated with
any type, due to annotator error.

Addressing is part of the dialogue-act annotation and is deliberately
only annotated for these meetings: ES2008a,b; ES2009c,d; IS1000a;
IS1001a,b,c; IS1003b,d; IS1006b,d; IS1008a,b,c,d; TS3005a,b
 

NOTES ON DATA FORMATS

Topic Segmentation note: for scenario meetings, topics point into a
type ontology of topic names, but where the annotator introduced a
non-standard topic title, the pointer points to 'other' in the
ontology and the attribute 'other_description' is filled in. For
non-scenarion meetings, the pointers are never present, and the
attribute 'other_description' is always filled in.

