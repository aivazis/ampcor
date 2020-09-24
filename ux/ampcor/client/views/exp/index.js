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
const exp = (props) => {
    // notify the store about the page flip
    props.flipPage('exp', 'the ampcor workflow')
    // build the container and return it
    return (
        <section style={styles.exp}>
            <div style={styles.viewport}>
                <div style={styles.plot}>
                    {Array(40*80).fill().map( (x,i) => (
                         <img src={`/exp/tile-${i}`} style={styles.tile} loading="lazy" />
                     ))}
                </div>
            </div>
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
export default connect(store, actions)(exp)


// end of file
