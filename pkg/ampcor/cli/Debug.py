# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2025 all rights reserved
#


# externals
import ampcor


# declaration
class Debug(ampcor.shells.command, family='ampcor.cli.debug'):
    """
    Display debugging information about this application
    """


    # user configurable state
    prefix = ampcor.properties.str()
    prefix.tip = "specify the portion of the namespace to display"


    @ampcor.export(tip="dump the application configuration namespace")
    def nfs(self, plexus, **kwds):
        """
        Dump the application configuration namespace
        """
        # get the prefix
        prefix = self.prefix or "ampcor"
        # show me
        plexus.pyre_nameserver.dump(prefix)
        # all done
        return 0


    @ampcor.export(tip="dump the application configuration namespace")
    def vfs(self, plexus, **kwds):
        """
        Dump the application virtual filesystem
        """
        # get the prefix
        prefix = self.prefix or '/ampcor'
        # build the report
        report = '\n'.join(plexus.vfs[prefix].dump())
        # sign in
        plexus.info.line('vfs: prefix={!r}'.format(prefix))
        # dump
        plexus.info.log(report)
        # all done
        return 0


# end of file
