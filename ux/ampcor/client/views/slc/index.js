// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React, { useState } from 'react'
// locals
import styles from './styles'
import { useFlowContext } from '~/context'
import { SLC } from '~/panels'


// explore the input SLCs
const slc = (props) => {
    // get the flow configuration
    const config = useFlowContext()
    // unpack the raster shapes
    const rasterShapes = { ref: config.reference.shape, sec: config.secondary.shape }

    // select the slc to show
    const [slc, setSLC] = useState("ref")
    // extract its shape
    const shape = rasterShapes[slc]

    // build the tile uri stem
    const uri = `/slc/${slc}`

    // build the container, fill it, and return it
    return (
        <section style={styles.slc}>
            <SLC uri={uri} shape={shape} style={styles?.slcPanel} />
        </section>
    )
}


// publish
export default slc


// end of file
