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
const slc = (props) => {
    // build the container and return it
    return (
        <section style={styles.slc}>
            <div style={styles.viewport}>
                <div style={styles.plot}>
                    {Array(40*80).fill().map( (x,i) => (
                         <img key={`tile-${i}`}
                              loading="lazy"
                              src={`/slc/ref/0/tile-${i}`} style={styles.tile}
                         />
                     ))}
                </div>
            </div>
        </section>
    )
}


// publish
export default slc


// end of file
