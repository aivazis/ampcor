// -*- web -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2020 all rights reserved
//


// a color wheel
const wheel = {
    // greys; the names are borrowed from {omni graffle}
    obsidian: "#000",
    basalt: "#333333",
    gabro: "#424242",
    steel: "#666",
    shale: "#686868",
    flint: "#8a8a8a",
    granite: "#9a9a9a",
    aluminum: "#a5a5a5",
    concrete: "#b8b8b8",
    soapstone: "#d6d6d6",
    cement: "#eee",
    marble: "#f1f1f1",
    flour: "#fafafa",
    chalk: "#ffffff",
}


// my dark theme
const dark = {
    // the page
    page: {
        background: "hsl(0deg, 0%, 7%)",
        appversion: "hsl(0deg, 0%, 25%)",
    },

    // the header
    banner: {
        // overall styling
        background: "hsl(0deg, 0%, 7%)",
        separator: "hsl(0deg, 0%, 15%)",
        // contents
        name: "hsl(28deg, 90%, 55%)",
    },

    // the footer
    colophon: {
        // overall styling
        background: "hsl(0deg, 0%, 7%)",
        separator: "hsl(0deg, 0%, 15%)",
        // contents
        copyright: "hsl(0deg, 0%, 30%)",
        author: "hsl(0deg, 0%, 40%)",
    },

    // widgets
    widgets: {
        background: "hsl(0deg, 0%, 7%)",
    },

    // journal colors
    journal: {
        error: "hsl(0deg, 90%, 50%)",
    }
}


// my default theme
const theme = dark


// publish
export { wheel, dark, theme }


// end of file
