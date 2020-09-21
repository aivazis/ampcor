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


// explore the input SLCs
const slc = (props) => {
    // notify the store about the page flip
    props.flipPage('slc', 'the ampcor workflow')
    // build the container and return it
    return (
        <section style={styles.slc}>
            <div style={styles.placeholder}>the input rasters</div>
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
export default connect(store, actions)(slc)


// end of file