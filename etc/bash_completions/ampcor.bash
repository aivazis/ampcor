# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2020 all rights reserved
#

# bash completion script for ampcor

function _ampcor() {
    # get the partial command line
    local line=${COMP_LINE}
    local word=${COMP_WORDS[COMP_CWORD]}
    # ask ampcor to provide guesses
    COMPREPLY=($(ampcor complete --word="${word}" --line="${line}"))
}

# register the hook
complete -F _ampcor ampcor

# end of file
