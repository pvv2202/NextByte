import React from 'react'
import { useEffect, useState, useContext } from 'react'
import { useForm } from 'react-hook-form';
import { api_request } from '../api_request';
import LoginForm from '../components/auth/loginForm';
import { useNavigate} from 'react-router';
import { get_user } from '../api_request';
import { UserContext } from '../UserContext';
import RecipeCard from '../components/recipeCard';
import {motion, AnimatePresence} from 'framer-motion'

function RecipeBook() {
  const {user, setUser} = useContext(UserContext)
  const [userRecipes, setUserRecipes] = useState([])
  const [loading, setLoading] = useState(false)

  
  useEffect(()=> {
    async function fetch_recipes() {
        const recipes = await api_request(
          'recipes/saved-recipes',
          'GET'
        )
        console.log(recipes)
        setUserRecipes(recipes)
    }

    fetch_recipes()
  }, [])


  return (
    <div className='relative flex flex-col w-full min-h-screen gap-y-20 items-center bg-white'>
      <div className='flex justify-start gap-5 w-full p-6 flex-wrap'>
        {userRecipes.map((recipe, idx)=>(
          <RecipeCard
            key={recipe[0]}
            id={recipe[0]}
            title={recipe[1]}
            img={recipe[4]}
            date={recipe[5]}
          />
        ))}
      </div>     
    </div>
  )
}

export default RecipeBook