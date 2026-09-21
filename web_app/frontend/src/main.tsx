import { CssBaseline, ThemeProvider, createTheme } from '@mui/material'
import { createRoot } from 'react-dom/client'
import App from './App'

const theme = createTheme({
  palette: { mode: 'dark', primary: { main: '#74c7ad' }, secondary: { main: '#74c7ad' }, background: { default: '#111615', paper: '#1a201e' } },
  shape: { borderRadius: 8 },
  typography: { fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif', fontSize: 13, h5: { fontSize: '1.5rem', fontWeight: 700 }, button: { textTransform: 'none', fontWeight: 600 } },
  components: { MuiButton: { defaultProps: { disableElevation: true } }, MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' } } }, MuiCard: { styleOverrides: { root: { backgroundImage: 'none', border: '1px solid #273548' } } }, MuiTableCell: { styleOverrides: { root: { borderColor: '#2a3748' } } } },
})

createRoot(document.getElementById('root')!).render(<ThemeProvider theme={theme}><CssBaseline /><App /></ThemeProvider>)
