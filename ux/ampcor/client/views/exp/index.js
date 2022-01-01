// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2022 all rights reserved


// externals
import React from 'react'
// locals
import styles from './styles'
import { Tile } from '~/widgets'


// explore the input SLCs
const exp = (props) => {
    // the tile uri
    const uri = "/exp/tile-0"

    // build the container and return it
    return (
        <section style={styles.exp}>
            <div style={styles.viewport}>
                <div style={styles.canvas}>
                    <Tile key="00" uri={uri} style={styles.tile} />
                    <Tile key="01" uri={uri} style={styles.tile} />
                    <Tile key="10" uri={uri} style={styles.tile} />
                    <Tile key="11" uri={uri} style={styles.tile} />
                </div>
            </div>
        </section>
    )
}


// publish
export default exp


// end of file
