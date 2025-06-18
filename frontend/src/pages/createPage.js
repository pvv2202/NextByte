import React, { useState } from 'react'
import {useForm} from 'react-hook-form'
import RecipeForm from '../components/create/recipeForm.js'
import CreateDisplay from '../components/create/createDisplay.js'
import { TypeAnimation } from 'react-type-animation'
import { api_request } from '../api_request'


function CreatePage() {
  // use form is from a react library helpful for managing form validation and setup https://react-hook-form.com/get-started
  const {register, handleSubmit, reset} = useForm()
  const [generating, setGenerating] = useState(false)
  const [recipe, setRecipe] = useState(null)

  // this will eventually request the server that the model lives on
  const generateRecipe = async (data) => {
    setRecipe(null)
    setGenerating(true)
    reset();
    const recipeTitle = data['recipeTitle']
    const recipe = await api_request(
      'recipes/generate',
      'POST',
      {'Content-Type': 'application/json'},
      {recipeTitle:recipeTitle}
    )
    setGenerating(false)

    setRecipe(recipe['recipe'])
  }

  return (
    <div className='relative flex flex-col py-20 w-full min-h-screen gap-y-20 items-center bg-white'>
      <TypeAnimation
        sequence={[
            "Next Bite",
            2000,
            "NextByte",
            5000
          ]}
        wrapper="div"
        speed={10}
        cursor={false}
        className="text-4xl font-mono text-sky-950 text-shadow-lg"
      />
      <CreateDisplay
        generating={generating}
        recipe={recipe}
        setRecipe={setRecipe}
      />
      <RecipeForm
        register={register}
        handleSubmit={handleSubmit}
        onSubmit={generateRecipe} 
        generating={generating}
      />  
    </div>
  )
}

export default CreatePage