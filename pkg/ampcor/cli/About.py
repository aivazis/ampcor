# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2021 all rights reserved
#


# externals
import ampcor


# declaration
class About(ampcor.shells.command, family='ampcor.cli.about'):
    """
    Display information about this application
    """


    @ampcor.export(tip="print the copyright note")
    def copyright(self, plexus, **kwds):
        """
        Print the copyright note of the ampcor package
        """
        # show the copyright note
        plexus.info.log(ampcor.meta.copyright)
        # all done
        return


    @ampcor.export(tip="print out the acknowledgments")
    def credits(self, plexus, **kwds):
        """
        Print out the license and terms of use of the ampcor package
        """
        # make some space
        plexus.info.log(ampcor.meta.header)
        # all done
        return


    @ampcor.export(tip="print out the license and terms of use")
    def license(self, plexus, **kwds):
        """
        Print out the license and terms of use of the ampcor package
        """
        # make some space
        plexus.info.log(ampcor.meta.license)
        # all done
        return


    @ampcor.export(tip="print the version number")
    def version(self, plexus, **kwds):
        """
        Print the version of the ampcor package
        """
        # make some space
        plexus.info.log(ampcor.meta.header)
        # all done
        return


# end of file
