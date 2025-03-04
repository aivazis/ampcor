#-*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2025 all rights reserved
#


# support
import ampcor


# declaration
class Config(ampcor.shells.command, family="ampcor.cli.config"):
    """
    Display configuration information about this package
    """


    # version info
    @ampcor.export(tip="the version information")
    def version(self, **kwds):
        """
        Print the version of the ampcor package
        """
        # print the version number
        print(f"{ampcor.meta.version}")
        # all done
        return 0


    # configuration
    @ampcor.export(tip="the top level installation directory")
    def prefix(self, **kwds):
        """
        Print the top level installation directory
        """
        # print the version number
        print(f"{ampcor.prefix}")
        # all done
        return 0


    @ampcor.export(tip="the directory with the executable scripts")
    def path(self, **kwds):
        """
        Print the location of the executable scripts
        """
        # print the version number
        print(f"{ampcor.prefix}/bin")
        # all done
        return 0


    @ampcor.export(tip="the directory with the python packages")
    def pythonpath(self, **kwds):
        """
        Print the directory with the python packages
        """
        # print the version number
        print(f"{ampcor.home.parent}")
        # all done
        return 0


    @ampcor.export(tip="the location of the {ampcor} headers")
    def incpath(self, **kwds):
        """
        Print the locations of the {ampcor} headers
        """
        # print the version number
        print(f"{ampcor.prefix}/include")
        # all done
        return 0


    @ampcor.export(tip="the location of the {ampcor} libraries")
    def libpath(self, **kwds):
        """
        Print the locations of the {ampcor} libraries
        """
        # print the version number
        print(f"{ampcor.prefix}/lib")
        # all done
        return 0



# end of file
