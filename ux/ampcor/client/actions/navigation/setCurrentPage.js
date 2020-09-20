// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// locals
import { NAVIGATION_CURRENT_PAGE } from './types'


// reducer
export default (page, title) => ({
    type: NAVIGATION_CURRENT_PAGE,
    payload: {page, title},
})


// end of file
