import '../App.css';
import { ClipLoader } from 'react-spinners';
import Login from './Login';
import { Navigate, Route, Routes, useNavigate } from 'react-router';
import SignUp from './SignUp';
import LandingPage from './landingPage';
import CreatePage from './createPage';
import RecipeBook from './recipeBook';
import Redirect from './redirect';
import { useState, useContext, useEffect, createContext} from 'react';
import { get_user } from '../api_request';
import { UserContext } from '../UserContext';
import FullRecipe from './fullRecipe.js';



/* First functional component */ 
function App() {
  const navigate = useNavigate()
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(()=>{
      async function fetch_user() {
        setLoading(true)
        const user = await get_user()
        setUser(user) 
        if (!user) navigate('/login')
        setLoading(false)
      }
      fetch_user()
      
  }, [])
  
  if(loading){
    return (
        <div className='relative flex flex-col w-full min-h-screen gap-y-20 items-center bg-white'>
          <ClipLoader color={'black'} speedMultiplier={0.5} size={100} />
        </div>
      )
  } 
  
  return (
    <UserContext.Provider value={{user, setUser, loading, setLoading}} >
   
      <div className="app flex w-screen min-h-screen bg-green-200 p-0">
        <Routes>
          <Route path='/' element={<Navigate to='/signup' />} />
          <Route path='/login' element={<Login/>} />
          <Route path='/signup' element={<SignUp />} />
          <Route path='/landing' element={<LandingPage />}>
              <Route path='' element={<CreatePage />} />
              <Route path='my-recipes' element={<RecipeBook />}/>
              <Route path='my-recipes/:recipe_id' element={<FullRecipe />}/> 
          </Route>
        </Routes> 
      </div>
    </UserContext.Provider>
  );
}

export default App;
