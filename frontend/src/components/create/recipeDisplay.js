import { useState, useEffect } from 'react';
import { TypeAnimation } from 'react-type-animation'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBookmark as faBookmarkSolid } from '@fortawesome/free-solid-svg-icons';
import { faBookmark as faBookmarkRegular } from '@fortawesome/free-regular-svg-icons';
import {motion, useAnimate, stagger, AnimatePresence} from 'framer-motion'
import { api_request } from '../../api_request';

function RecipeDisplay({recipe, setRecipe}) {
  const [saved, setSaved] = useState(false)
  
  let title = '';
  let ingredients = [];
  let directions = [];

  if (recipe) {
    title = recipe['title'];
    ingredients = recipe['ingredients'];
    directions = recipe['directions'];
  }

  const saveRecipe = async () => {
    console.log('saving recipe')
    setSaved(true)
    const response = await api_request(
      'recipes/saved-recipes',
      'POST',
      {'Content-Type': 'application/json'},
      {
        title: title,
        ingredients: ingredients.join("||"),
        directions: directions.join("||")
      }
    )
    setRecipe(null)
  } 
  
  
  return (
    <AnimatePresence>
        <motion.div 
          className='w-full flex flex-col bg-white p-8 rounded-xl shadow-md'
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.3}}
          exit={{ scale: 0 }}
            
        >
          <div className='flex gap-1 justify-between'>
            <h2 className='text-2xl'>
              {title.split(" ").map((el, i) => (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{
                    duration: 0.5,
                    delay: i / 10
                  }}
                  key={i}
                  >
                  {el}{" "}
                  </motion.span>
              ))}
            </h2>
            <motion.button 
              whileHover={{ scale: 1.1 }} 
              whileTap={{ scale: 0.9 }}  
              onClick={() => saveRecipe()}>
              {saved ? <FontAwesomeIcon icon={faBookmarkSolid} /> : <FontAwesomeIcon icon={faBookmarkRegular}/> }  
            </motion.button>
          </div> 

          <ul className='ml-4 mt-2 list-disc'>
            {ingredients.map((ing, idx)=>(
              <motion.li key={idx}>
                {ing.split(" ").map((el, i) => (
                  <motion.span
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{
                      duration: 0.5,
                      delay: i / 10
                    }}
                    key={i}
                    >
                    {el}{" "}
                    </motion.span>
              ))}
              </motion.li>
            ))}
          </ul>

          <ol className='ml-4 mt-2 list-decimal'>
              {directions.map((dir, idx)=>(
                <motion.li key={idx}>
                  {dir.split(" ").map((el, i) => (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{
                        duration: 0.5,
                        delay: i / 10
                      }}
                      key={i}
                      >
                      {el}{" "}
                      </motion.span>
                  ))}
                </motion.li>
              ))}
            </ol>

            <div className='flex justify-end'>
              {recipe !== null ? 
                <motion.button 
                whileHover={{ scale: 1.1 }} 
                whileTap={{ scale: 0.9 }} 
                className='w-1/4 mt-2 hover:scale-105 text-white rounded bg-gray-800' 
                onClick={() => setRecipe(null)}>
                  Clear
                </motion.button> : 
                ''}
            </div>

      </motion.div>
    </AnimatePresence>
    
  )
}

export default RecipeDisplay










                   