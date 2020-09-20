// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import immutable from 'immutable'

// support
import {
    NAVIGATION_CURRENT_PAGE
} from '~/actions/navigation/types'


// initial state
const empty = immutable.Map({
    page: '',
    title: '',
})


// the reducer
export default (navigation=empty, {type, payload}) => {
    // set the current page metadata
    if (type == NAVIGATION_CURRENT_PAGE) {
        // unpack
        const {page, title} = payload
        // update the metadata
        navigation = navigation.withMutations(map => {
            map.set('page', page).set('title', title)
        })
        // all done
        return navigation
    }

    // if we get this far, we didn't recognize the action; do nothing to the store
    return navigation
}


// end of file
