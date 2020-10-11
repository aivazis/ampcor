// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved


// externals
import React from 'react'


// create a context for maintaining the mosaic zoom level
const Context = React.createContext({
    // the reference to the mosaic
    mosaicRef: null,

    // the current zoom level
    zoom: 0,
    // the minimum zoom level
    minZoom: 0,
    // the maximum zoom level
    maxZoom: Infinity,

    // if the client doesn't set up a context, setting the zoom level makes me
    setZoom() {
        // complain
        throw new Error("setZoom: no context provider")
    },
})


// the mosaic context provider factory
const Provider = ({mosaicRef, minZoom=0, maxZoom=Infinity, children}) => {
    // build the current {zoom} level state
    const [zoom, setZoom] = React.useState(maxZoom)
    // initialize the context and make it available to my {children}
    return (
        <Context.Provider value={{zoom, setZoom, maxZoom, minZoom, mosaicRef}}>
            {children}
        </Context.Provider>
    )
}


// hook that provides access to the current zoom level and its mutators
const useZoom = () => {
    //
    const {zoom, setZoom, minZoom, maxZoom, mosaicRef} = React.useContext(Context)

    // build the callbacks that tie the zoom controls to my state
    // zoom in is clipped at {minZoom}
    const zoomIn = () => (zoom > minZoom ? setZoom(zoom-1) : null)
    // zoom out is clipped at {maxZoom}
    const zoomOut = () => (zoom < maxZoom ? setZoom(zoom+1) : null)

    // and publish
    return { zoom, zoomIn, zoomOut }
}


// publish
export default { Context, Provider, useZoom }


// end of file
