// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2022 all rights reserved


// externals
import React, { useState } from 'react'
// locals
import styles from './style'


// currently, a tile is an {img} that's lazy loaded from a {uri}
const widget = ({uri, style}) => {
    // mix the styling
    const tileStyle = { ...styles, ...style }
    // render
    return (
        <img loading="lazy" className="lazyload" data-src={uri}
             style={tileStyle}
        />
    )
}


// publish
export default widget


// end of file
