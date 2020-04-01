const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin'); //installed via npm
const webpack = require('webpack'); //to access built-in plugins

const {
    NODE_ENV = 'production'
} = process.env;

module.exports = {
    mode: NODE_ENV,
    entry: path.join(__dirname, 'src/index.js'),
    output: {
        path: path.resolve(__dirname, 'dist')
    },
    devServer: {
        contentBase: path.join(__dirname, 'dist'),
        compress: true,
    },
    plugins: [
        new HtmlWebpackPlugin({template: './src/index.html'})
    ],
    watch: NODE_ENV === 'development'
}