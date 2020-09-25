// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React from 'react'
// locals
import styles from './styles'


// explore the input SLCs
const exp = (props) => {
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


// publish
export default exp


// end of file
