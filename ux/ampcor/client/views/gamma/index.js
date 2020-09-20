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
const gamma = (props) => {
    // notify the store about the page flip
    props.flipPage('gamma', 'the correlation surface')
    // the container
    return (
        <section style={styles.gamma}>
            <div style={styles.placeholder}>the correlation surface</div>
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
export default connect(store, actions)(gamma)


// end of file
