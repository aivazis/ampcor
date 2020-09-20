// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React from 'react'
import { connect } from 'react-redux'
// locals
import styles from './styles'


// the area
const offsets = (props) => {
    // notify the store about the page flip
    props.flipPage('offsets', 'the estimated offsets')
    // build the container and return it
    return (
        // the container
        <section style={styles.offsets}>
            <div style={styles.placeholder}>the offsets</div>
        </section>
    )
}


// store access
const store = null

// actions
import { setCurrentPage } from '~/actions/navigation'
// dispatch
const actions = (dispatch) => ({
    // navigational
    flipPage: (page, title) => dispatch(setCurrentPage(page, title)),
})


// connect to the state store and publish
export default connect(store, actions)(offsets)


// end of file
